import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from data.dataset import build_dataloader
from losses.structure_losses import inter_intra_distance_metrics
from transforms.frequency import batch_frequency_transform
from utils.metrics import diagnosis_metrics, frequency_metrics, separability_metrics
from utils.plotting import save_confusion_matrix, save_latent_pca
from utils.save_utils import ensure_dir, load_config, print_config
from utils.seed import set_seed


def device_from(config):
    return torch.device("cuda" if config.get("device", "auto") == "auto" and torch.cuda.is_available() else config.get("device", "cpu"))


@torch.no_grad()
def collect_spectra(config, split, device):
    loader = build_dataloader(config, split, shuffle=False)
    specs, labels = [], []
    for batch in loader:
        x = batch["signal"].to(device).float()
        s = batch_frequency_transform(x, **config["frequency"]).squeeze(1)
        specs.append(s.cpu().numpy())
        if "label" in batch:
            labels.extend(batch["label"].numpy().tolist())
    y = np.asarray(labels, dtype=np.int64) if labels else None
    return np.concatenate(specs, axis=0).astype(np.float32), y


def fit_pod(train_spectrum, standardize=False):
    mu = train_spectrum.mean(axis=0, keepdims=True)
    centered = train_spectrum - mu
    scale = centered.std(axis=0, keepdims=True).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    x = centered / scale if standardize else centered
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    energy = singular_values**2
    cumulative = np.cumsum(energy) / np.sum(energy)
    return mu.astype(np.float32), scale.astype(np.float32), vt.astype(np.float32), singular_values.astype(np.float32), cumulative.astype(np.float32)


def project_reconstruct(spectrum, mu, scale, basis, standardize=False):
    centered = spectrum - mu
    x = centered / scale if standardize else centered
    coeff = x @ basis.T
    recon_x = coeff @ basis
    recon = recon_x * scale + mu if standardize else recon_x + mu
    return coeff.astype(np.float32), recon.astype(np.float32)


def spectral_reconstruction_metrics(clean, recon):
    fm = frequency_metrics(clean[:, None, :], recon[:, None, :])
    return {
        "spectral_rmse": float(np.sqrt(np.mean((recon - clean) ** 2))),
        "spectral_mae": float(np.mean(np.abs(recon - clean))),
        "spectral_corr": fm["envelope_corr"],
        "envelope_corr": fm["envelope_corr"],
        "fpcc": fm["fpcc"],
    }


def write_csv(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path.resolve()}")


def plot_energy(ranks, energy, out):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip energy plot: {exc}")
        return
    ensure_dir(Path(out).parent)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(energy) + 1), energy, linewidth=1.5)
    plt.scatter(ranks, [energy[r - 1] for r in ranks], c="tab:red", s=28)
    plt.axhline(0.95, color="k", linestyle="--", linewidth=1)
    plt.xlabel("POD rank")
    plt.ylabel("cumulative energy ratio")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Saved: {Path(out).resolve()}")


def plot_metrics(summary_csv, out, metrics):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"Skip metrics plot: {exc}")
        return
    df = pd.read_csv(summary_csv)
    df = df[df["split"] == "test"].copy()
    if df.empty:
        return
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 2.6 * len(metrics)), squeeze=False)
    for ax, metric in zip(axes[:, 0], metrics):
        ax.plot(df["rank"], df[metric], marker="o")
        ax.set_xlabel("POD rank")
        ax.set_ylabel(metric)
    fig.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {Path(out).resolve()}")


def plot_basis(basis, out, max_basis=8):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip basis plot: {exc}")
        return
    n = min(max_basis, len(basis))
    fig, axes = plt.subplots(n, 1, figsize=(10, max(2, 1.4 * n)), squeeze=False)
    for i in range(n):
        axes[i, 0].plot(basis[i], linewidth=1)
        axes[i, 0].set_title(f"POD basis {i + 1}")
    fig.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {Path(out).resolve()}")


def evaluate_rank(rank, packs, labels, mu, scale, vt, energy_ratio, out_dir, standardize):
    basis = vt[:rank]
    np.savez(
        out_dir / f"pod_basis_rank_{rank}.npz",
        mean=mu.squeeze(0),
        scale=scale.squeeze(0),
        basis=basis,
        singular_values=None,
        rank=rank,
        energy_ratio=energy_ratio,
        standardize=standardize,
    )
    rows = []
    coeffs, recons = {}, {}
    for split, spectrum in packs.items():
        coeff, recon = project_reconstruct(spectrum, mu, scale, basis, standardize)
        coeffs[split], recons[split] = coeff, recon
        row = {"rank": rank, "split": split, "pod_energy_ratio": float(energy_ratio), **spectral_reconstruction_metrics(spectrum, recon)}
        y = labels.get(split)
        if y is not None:
            row.update(inter_intra_distance_metrics(coeff, y))
            sep = separability_metrics(coeff, y)
            row.update({
                "silhouette": sep.get("silhouette", np.nan),
                "davies_bouldin": sep.get("davies_bouldin", np.nan),
                "calinski_harabasz": sep.get("calinski_harabasz", np.nan),
            })
        rows.append(row)
    if labels.get("train") is not None and labels.get("test") is not None:
        diag, cm = diagnosis_metrics(coeffs["train"], labels["train"], coeffs["test"], labels["test"])
        for row in rows:
            if row["split"] == "test":
                row.update(diag)
        save_confusion_matrix(cm, out_dir / f"confusion_matrix_rank_{rank}.png", title=f"POD coeff rank {rank}")
    return rows, coeffs, recons


def clean_feature_baseline(packs, labels):
    if labels.get("train") is None or labels.get("test") is None:
        return {}
    diag, _ = diagnosis_metrics(packs["train"], labels["train"], packs["test"], labels["test"])
    mani = inter_intra_distance_metrics(packs["test"], labels["test"])
    sep = separability_metrics(packs["test"], labels["test"])
    return {
        **diag,
        **mani,
        "silhouette": sep.get("silhouette", np.nan),
        "davies_bouldin": sep.get("davies_bouldin", np.nan),
        "calinski_harabasz": sep.get("calinski_harabasz", np.nan),
    }


def write_summary(out_dir, ranks, energy, test_rows, clean_diag):
    def first_rank(predicate):
        for row in test_rows:
            if predicate(row):
                return int(row["rank"])
        return None

    r95 = next((r for r in range(1, len(energy) + 1) if energy[r - 1] >= 0.95), None)
    env_rank = first_rank(lambda r: r["envelope_corr"] >= 0.98)
    best_log = max(test_rows, key=lambda r: r.get("logistic_accuracy", -1))
    best_env = max(test_rows, key=lambda r: r["envelope_corr"])
    recommended = first_rank(lambda r: r["pod_energy_ratio"] >= 0.95 and r["envelope_corr"] >= 0.98)
    if recommended is None:
        recommended = int(best_env["rank"])
    clean_log = clean_diag.get("logistic_accuracy", np.nan)
    pod_log = best_log.get("logistic_accuracy", np.nan)
    feasible = (
        recommended is not None
        and recommended <= 64
        and best_env["envelope_corr"] >= 0.98
        and (np.isnan(clean_log) or pod_log >= clean_log - 0.05)
    )
    lines = [
        "POD upper-bound analysis",
        "",
        "Energy ratio by requested rank:",
    ]
    for r in ranks:
        lines.append(f"- rank {r}: {energy[r - 1]:.6f}")
    lines.extend(
        [
            "",
            f"First rank reaching 95% energy: {r95}",
            f"First requested rank with envelope_corr >= 0.98 on test: {env_rank}",
            f"Best test envelope_corr: rank {int(best_env['rank'])}, {best_env['envelope_corr']:.6f}",
            f"Clean spectrum Logistic acc baseline: {clean_log:.6f}",
            f"Best POD coeff Logistic acc: rank {int(best_log['rank'])}, {pod_log:.6f}",
            f"Recommended POD_RANK: {recommended}",
            f"Continue to POD coefficient predictor / POD-FMED-Net: {'yes' if feasible else 'no'}",
            "",
            "Decision: POD basis is treated as feasible only when a low rank preserves spectral reconstruction and does not materially reduce diagnosis separability.",
        ]
    )
    path = out_dir / "pod_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {path.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    config = load_config(args.config)
    print_config(config)
    set_seed(config["seed"])
    device = device_from(config)
    out_dir = ensure_dir(config["paths"]["pod_analysis_dir"])

    packs, labels = {}, {}
    for split in ["train", "val", "test"]:
        spectrum, y = collect_spectra(config, split, device)
        packs[split], labels[split] = spectrum, y
        print(f"{split}: spectrum={spectrum.shape}, labels={None if y is None else y.shape}")
    if labels["train"] is not None:
        for split in ["val", "test"]:
            if labels[split] is None or set(np.unique(labels[split])) != set(np.unique(labels["train"])):
                raise RuntimeError(f"Label classes in {split} do not match train.")

    mu, scale, vt, singular_values, energy = fit_pod(packs["train"], config["pod"].get("standardize", False))
    rank_list = [r for r in config["pod"]["rank_list"] if r <= vt.shape[0] and r <= vt.shape[1]]
    if not rank_list:
        raise RuntimeError(f"No valid POD ranks for basis shape {vt.shape}.")
    np.savez(out_dir / "pod_svd_full.npz", mean=mu.squeeze(0), scale=scale.squeeze(0), basis=vt, singular_values=singular_values, energy_ratio=energy)

    all_rows = []
    coeff_for_plot = None
    for rank in rank_list:
        rows, coeffs, _ = evaluate_rank(rank, packs, labels, mu, scale, vt, energy[rank - 1], out_dir, config["pod"].get("standardize", False))
        all_rows.extend(rows)
        if rank == config["pod"].get("rank", rank_list[0]):
            coeff_for_plot = coeffs
    clean_diag = clean_feature_baseline(packs, labels)
    clean_row = {"rank": 0, "split": "test", "feature": "clean_spectrum", **clean_diag}
    if clean_diag:
        all_rows.append(clean_row)
    summary_csv = out_dir / "pod_upper_bound_summary.csv"
    write_csv(summary_csv, all_rows)
    plot_energy(rank_list, energy, out_dir / "pod_energy_curve.png")
    plot_metrics(summary_csv, out_dir / "pod_reconstruction_metrics.png", ["pod_energy_ratio", "spectral_rmse", "envelope_corr"])
    plot_metrics(summary_csv, out_dir / "pod_diagnosis_metrics.png", ["knn_accuracy", "logistic_accuracy", "inter_intra_ratio"])
    plot_basis(vt[: min(max(rank_list), 16)], out_dir / "pod_basis_visualization.png")
    if coeff_for_plot is not None and labels.get("test") is not None:
        save_latent_pca(coeff_for_plot["test"], labels["test"], out_dir / "pod_coeff_pca.png", title=f"POD coeff PCA rank {config['pod'].get('rank')}")
    test_rows = [r for r in all_rows if r.get("split") == "test" and r.get("rank", 0) != 0]
    write_summary(out_dir, rank_list, energy, test_rows, clean_diag)


if __name__ == "__main__":
    main()
