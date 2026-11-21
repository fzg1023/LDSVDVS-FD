import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import build_dataloader
from losses.structure_losses import supervised_contrastive_loss
from models.denoiser_rescnn_tcn import ResCNN1DDenoiser
from models.pod_denoisers import UWNO1DDenoiser
from models.unet_baseline import UNet1DBaseline
from pod_upper_bound_analysis import fit_pod
from run_pod_denoising_experiments import (
    PODProjector,
    collect_clean_arrays,
    denormalize_batch,
    device_from,
    make_noisy_for_model,
)
from transforms.frequency import batch_frequency_transform
from utils.metrics import diagnosis_metrics, frequency_metrics, waveform_metrics
from utils.save_utils import ensure_dir, load_config, save_json
from utils.seed import set_seed


class LatentDenoiser(nn.Module):
    def __init__(self, rank, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rank, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, rank),
        )

    def forward(self, coeff):
        return self.net(coeff)


def write_csv(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path.resolve()}")


def build_backbone(name, config, device):
    base = int(config["model"]["base_channels"])
    if name == "rescnn":
        return ResCNN1DDenoiser(width=64, depth=8, kernel_size=7, residual=False).to(device)
    if name == "uwno":
        return UWNO1DDenoiser(base_channels=base, width=128, depth=3, levels=3, residual=False).to(device)
    if name == "unet":
        return UNet1DBaseline(base_channels=base).to(device)
    raise ValueError(f"Unknown backbone: {name}")


def ensure_pod(config, device):
    rank = int(config["pod"]["rank"])
    path = Path(config["paths"]["pod_analysis_dir"]) / f"pod_basis_rank_{rank}.npz"
    if path.exists():
        data = np.load(path)
        return {"mean": data["mean"].astype(np.float32), "basis": data["basis"].astype(np.float32), "rank": rank}
    _, train_spec, _ = collect_clean_arrays(config, "train", device)
    mu, _, vt, _, energy = fit_pod(train_spec, config["pod"].get("standardize", False))
    ensure_dir(path.parent)
    np.savez(path, mean=mu.squeeze(0), basis=vt[:rank], rank=rank, energy_ratio=energy[rank - 1])
    return {"mean": mu.squeeze(0).astype(np.float32), "basis": vt[:rank].astype(np.float32), "rank": rank}


def pod_coeff_and_recon(pod, spectrum):
    coeff = pod.coeff(spectrum)
    recon = pod.reconstruct(coeff)
    return coeff, recon


def build_clean_pod_centers(config, pod, device):
    _, train_spec, train_labels = collect_clean_arrays(config, "train", device)
    coeff = ((train_spec - pod.mean.cpu().numpy()) @ pod.basis.cpu().numpy().T).astype(np.float32)
    labels = np.asarray(train_labels, dtype=np.int64)
    classes = np.asarray(sorted(np.unique(labels)), dtype=np.int64)
    centers = []
    for cls in classes:
        centers.append(coeff[labels == cls].mean(axis=0))
    return {
        "classes": torch.as_tensor(classes, dtype=torch.long, device=device),
        "centers": torch.as_tensor(np.stack(centers), dtype=torch.float32, device=device),
    }


def label_to_center_index(labels, classes):
    labels = labels.long()
    if torch.equal(classes, torch.arange(len(classes), device=classes.device)):
        return labels
    idx = torch.empty_like(labels)
    for i, cls in enumerate(classes):
        idx[labels == cls] = i
    return idx


def fixed_center_separation_loss(coeff, labels, center_data, margin_cos=0.2):
    if labels is None or center_data is None:
        return coeff.new_tensor(0.0)
    classes = center_data["classes"].to(coeff.device)
    centers = center_data["centers"].to(coeff.device)
    idx = label_to_center_index(labels.to(coeff.device), classes)
    z = F.normalize(coeff, dim=1)
    c = F.normalize(centers, dim=1)
    positive = F.mse_loss(z, c[idx])
    cosine = z @ c.T
    neg_mask = torch.ones_like(cosine, dtype=torch.bool)
    neg_mask[torch.arange(len(idx), device=coeff.device), idx] = False
    negative = F.relu(cosine[neg_mask] - margin_cos).pow(2).mean()
    return positive + negative


def train_one(backbone_name, method, config, pod, device, out_dir, center_data=None):
    model = build_backbone(backbone_name, config, device)
    latent = LatentDenoiser(config["pod"]["rank"]).to(device) if method == "latent_denoising" else None
    params = list(model.parameters()) + (list(latent.parameters()) if latent is not None else [])
    opt = torch.optim.AdamW(params, lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    train_loader = build_dataloader(config, "train", shuffle=True)
    val_loader = build_dataloader(config, "val", shuffle=False)
    rows = []
    best = {"score": -1e18}
    snrs = config["data"]["train_snr_choices"]
    weights = config["loss_weights"]

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        if latent is not None:
            latent.train()
        sums = {"total": 0.0, "rec": 0.0, "freq": 0.0, "pod": 0.0, "cons": 0.0, "latent": 0.0, "supcon": 0.0, "center": 0.0}
        steps = 0
        for bi, batch in enumerate(train_loader):
            clean_raw = batch["signal"].to(device).float()
            clean_model, noisy_model, noisy_raw, norm_meta = make_noisy_for_model(clean_raw, snrs, config, bi * 1000)
            pred_model, _ = model(noisy_model, return_aux=True)
            pred_raw = denormalize_batch(pred_model, norm_meta)
            s_clean = batch_frequency_transform(clean_raw, **config["frequency"])
            s_pred = batch_frequency_transform(pred_raw, **config["frequency"])
            loss_rec = F.l1_loss(pred_model, clean_model)
            loss_freq = F.l1_loss(s_pred, s_clean)
            loss_pod = clean_raw.new_tensor(0.0)
            loss_cons = clean_raw.new_tensor(0.0)
            loss_latent = clean_raw.new_tensor(0.0)
            loss_supcon = clean_raw.new_tensor(0.0)
            loss_center = clean_raw.new_tensor(0.0)

            if method == "output_constraint":
                c_clean, _ = pod_coeff_and_recon(pod, s_clean)
                c_pred, s_pred_pod = pod_coeff_and_recon(pod, s_pred)
                loss_pod = F.mse_loss(c_pred, c_clean) + F.l1_loss(s_pred_pod, s_clean)
            elif method == "env_supcon":
                labels = batch.get("label")
                if labels is not None:
                    labels = labels.to(device)
                c_pred, _ = pod_coeff_and_recon(pod, s_pred)
                loss_supcon = supervised_contrastive_loss(c_pred, labels, temperature=0.2)
            elif method == "align_center":
                labels = batch.get("label")
                if labels is not None:
                    labels = labels.to(device)
                c_clean, _ = pod_coeff_and_recon(pod, s_clean)
                c_pred, s_pred_pod = pod_coeff_and_recon(pod, s_pred)
                loss_pod = F.mse_loss(c_pred, c_clean) + 0.5 * F.l1_loss(s_pred_pod, s_clean)
                loss_center = fixed_center_separation_loss(c_pred, labels, center_data)
            elif method == "latent_denoising":
                c_clean, _ = pod_coeff_and_recon(pod, s_clean)
                c_noisy, _ = pod_coeff_and_recon(pod, batch_frequency_transform(noisy_raw, **config["frequency"]))
                c_lat = latent(c_noisy)
                s_lat = pod.reconstruct(c_lat)
                loss_latent = F.mse_loss(c_lat, c_clean) + F.l1_loss(s_lat, s_clean)
            elif method == "consistency":
                _, noisy_model_2, _, norm_meta_2 = make_noisy_for_model(clean_raw, snrs, config, bi * 1000 + 11_111)
                pred_model_2, _ = model(noisy_model_2, return_aux=True)
                pred_raw_2 = denormalize_batch(pred_model_2, norm_meta_2)
                c1, _ = pod_coeff_and_recon(pod, s_pred)
                c2, _ = pod_coeff_and_recon(pod, batch_frequency_transform(pred_raw_2, **config["frequency"]))
                loss_cons = F.mse_loss(c1, c2)
            elif method not in {"postprocess", "baseline"}:
                raise ValueError(f"Unknown method: {method}")

            loss = (
                weights["lambda_rec"] * loss_rec
                + weights["lambda_freq"] * loss_freq
                + weights.get("lambda_output_pod", 0.05) * loss_pod
                + weights.get("lambda_consistency", 0.02) * loss_cons
                + weights.get("lambda_latent_denoising", 0.05) * loss_latent
                + weights.get("lambda_supcon", 0.01) * loss_supcon
                + weights.get("lambda_center", 0.02) * loss_center
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if config["training"].get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(params, config["training"]["grad_clip"])
            opt.step()
            sums["total"] += float(loss.detach().cpu())
            sums["rec"] += float(loss_rec.detach().cpu())
            sums["freq"] += float(loss_freq.detach().cpu())
            sums["pod"] += float(loss_pod.detach().cpu())
            sums["cons"] += float(loss_cons.detach().cpu())
            sums["latent"] += float(loss_latent.detach().cpu())
            sums["supcon"] += float(loss_supcon.detach().cpu())
            sums["center"] += float(loss_center.detach().cpu())
            steps += 1

        val = evaluate(backbone_name, method, config, model, latent, pod, val_loader, device, snr=-4, split="val")
        score = val["snr_improvement"] + 0.5 * val["envelope_corr_eval"] - 0.5 * val["rmse"]
        row = {"epoch": epoch, "score": score, **{f"train_{k}": v / max(1, steps) for k, v in sums.items()}, **val}
        rows.append(row)
        if score > best["score"]:
            best["score"] = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "latent": latent.state_dict() if latent is not None else None,
                    "config": config,
                    "backbone": backbone_name,
                    "method": method,
                    "epoch": epoch,
                },
                out_dir / "best.pth",
            )
        print(
            f"{backbone_name}/{method} epoch {epoch}: "
            f"loss={row['train_total']:.5f} snr={val['snr_improvement']:.3f} "
            f"rmse={val['rmse']:.4f} env_eval={val['envelope_corr_eval']:.3f}"
        )
        if epoch % 10 == 0:
            write_csv(out_dir / "train_val_log.csv", rows)

    write_csv(out_dir / "train_val_log.csv", rows)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "config": config,
                "backbone": backbone_name, "method": method, "epoch": int(config["training"]["epochs"])},
               out_dir / "last_epoch.pth")
    state = torch.load(out_dir / "best.pth", map_location=device)
    model.load_state_dict(state["model"])
    if latent is not None and state["latent"] is not None:
        latent.load_state_dict(state["latent"])
    torch.save({"model": model.state_dict(), "latent": latent.state_dict() if latent is not None else None}, out_dir / "last_loaded_best.pth")
    return model, latent


@torch.no_grad()
def collect_eval_pack(method, config, model, latent, pod, loader, device, snr):
    model.eval()
    if latent is not None:
        latent.eval()
    clean_all, noisy_all, pred_all = [], [], []
    s_clean_all, s_pred_all, s_eval_all, labels = [], [], [], []
    for bi, batch in enumerate(loader):
        clean_raw = batch["signal"].to(device).float()
        clean_model, noisy_model, noisy_raw, norm_meta = make_noisy_for_model(clean_raw, snr, config, bi * 1000)
        pred_model, _ = model(noisy_model, return_aux=True)
        pred_raw = denormalize_batch(pred_model, norm_meta)
        s_clean = batch_frequency_transform(clean_raw, **config["frequency"])
        s_pred = batch_frequency_transform(pred_raw, **config["frequency"])
        s_eval = s_pred
        if method == "postprocess":
            c_pred = pod.coeff(s_pred)
            s_eval = pod.reconstruct(c_pred)
        elif method == "latent_denoising":
            c_noisy = pod.coeff(batch_frequency_transform(noisy_raw, **config["frequency"]))
            s_eval = pod.reconstruct(latent(c_noisy))
        clean_all.append(clean_raw.cpu().numpy())
        noisy_all.append(noisy_raw.cpu().numpy())
        pred_all.append(pred_raw.cpu().numpy())
        s_clean_all.append(s_clean.cpu().numpy())
        s_pred_all.append(s_pred.cpu().numpy())
        s_eval_all.append(s_eval.cpu().numpy())
        if "label" in batch:
            labels.extend(batch["label"].numpy().tolist())
    return {
        "clean": np.concatenate(clean_all),
        "noisy": np.concatenate(noisy_all),
        "pred": np.concatenate(pred_all),
        "s_clean": np.concatenate(s_clean_all),
        "s_pred": np.concatenate(s_pred_all),
        "s_eval": np.concatenate(s_eval_all),
        "labels": np.asarray(labels) if labels else None,
    }


@torch.no_grad()
def evaluate(backbone_name, method, config, model, latent, pod, loader, device, snr, split):
    pack = collect_eval_pack(method, config, model, latent, pod, loader, device, snr)
    row = {
        "backbone": backbone_name,
        "method": method,
        "split": split,
        "snr_db": snr,
        **waveform_metrics(pack["clean"], pack["pred"], pack["noisy"]),
    }
    pred_freq = frequency_metrics(pack["s_clean"], pack["s_pred"])
    eval_freq = frequency_metrics(pack["s_clean"], pack["s_eval"])
    row.update({f"{k}_pred": v for k, v in pred_freq.items()})
    row.update({f"{k}_eval": v for k, v in eval_freq.items()})
    row["envelope_corr"] = row["envelope_corr_pred"]
    row["spectral_rmse"] = row["spectral_rmse_pred"]
    row["envelope_corr_eval"] = row["envelope_corr_eval"]
    row["spectral_rmse_eval"] = row["spectral_rmse_eval"]
    return row


@torch.no_grad()
def add_diagnosis_metrics(rows, config, model_map, pod, device):
    train_loader = build_dataloader(config, "train", shuffle=False)
    _, train_spec, train_labels = collect_clean_arrays(config, "train", device)
    train_coeff = ((train_spec - pod.mean.cpu().numpy()) @ pod.basis.cpu().numpy().T).astype(np.float32)
    for row in rows:
        key = (row["backbone"], row["method"])
        model, latent = model_map[key]
        test_loader = build_dataloader(config, "test", shuffle=False)
        pack = collect_eval_pack(row["method"], config, model, latent, pod, test_loader, device, row["snr_db"])
        test_spec = pack["s_eval"].reshape(len(pack["s_eval"]), -1)
        test_coeff = ((test_spec - pod.mean.cpu().numpy()) @ pod.basis.cpu().numpy().T).astype(np.float32)
        diag, _ = diagnosis_metrics(train_coeff, train_labels, test_coeff, pack["labels"])
        row.update({f"pod_{k}": v for k, v in diag.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pod_fmed_config_4096_rms.py")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--out", default="outputs_4096_norm_rms/pod_idea_tests_rank32")
    parser.add_argument("--backbones", nargs="*", default=["uwno", "rescnn"])
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["output_constraint", "postprocess", "latent_denoising", "consistency"],
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config["pod"]["rank"] = int(args.rank)
    config["training"]["epochs"] = int(args.epochs)
    config["data"]["train_snr_choices"] = [-6, -4, -2, 0]
    config["data"]["test_snr_list"] = [0, -2, -4, -6]
    config["paths"]["pod_denoising_ablation_dir"] = args.out
    config["loss_weights"] = {
        **config["loss_weights"],
        "lambda_output_pod": 0.05,
        "lambda_consistency": 0.02,
        "lambda_latent_denoising": 0.05,
        "lambda_supcon": 0.01,
        "lambda_center": 0.02,
    }
    set_seed(config["seed"])
    device = device_from(config)
    base = ensure_dir(args.out)
    save_json(config, base / "config.json")
    pod_data = ensure_pod(config, device)
    pod = PODProjector(pod_data["mean"], pod_data["basis"]).to(device)
    center_data = build_clean_pod_centers(config, pod, device)

    model_map = {}
    summary_rows, by_snr_rows = [], []
    for backbone in args.backbones:
        for method in args.methods:
            out_dir = ensure_dir(base / f"{backbone}_{method}")
            model, latent = train_one(backbone, method, config, pod, device, out_dir, center_data=center_data)
            model_map[(backbone, method)] = (model, latent)
            test_loader = build_dataloader(config, "test", shuffle=False)
            for snr in config["data"]["test_snr_list"]:
                by_snr_rows.append(evaluate(backbone, method, config, model, latent, pod, test_loader, device, snr, "test"))
            mean_row = {"backbone": backbone, "method": method}
            for key in by_snr_rows[-len(config["data"]["test_snr_list"])].keys():
                if key in {"backbone", "method", "split"}:
                    continue
                vals = [r[key] for r in by_snr_rows[-len(config["data"]["test_snr_list"]):] if isinstance(r.get(key), (int, float, np.floating))]
                if vals:
                    mean_row[f"mean_{key}"] = float(np.mean(vals))
            mean_row["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
            if latent is not None:
                mean_row["latent_params_M"] = sum(p.numel() for p in latent.parameters()) / 1e6
            summary_rows.append(mean_row)

    add_diagnosis_metrics(by_snr_rows, config, model_map, pod, device)
    # Average diagnosis after it has been added to the SNR rows.
    for mean_row in summary_rows:
        matched = [r for r in by_snr_rows if r["backbone"] == mean_row["backbone"] and r["method"] == mean_row["method"]]
        for key in matched[0].keys():
            if key.startswith("pod_"):
                mean_row[f"mean_{key}"] = float(np.mean([r[key] for r in matched]))

    write_csv(base / "pod_idea_test_by_snr.csv", by_snr_rows)
    write_csv(base / "pod_idea_test_summary.csv", summary_rows)
    print(f"Saved all results under: {base.resolve()}")


if __name__ == "__main__":
    main()
