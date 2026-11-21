import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import build_dataloader
from data.noise import add_awgn
from losses.structure_losses import graph_manifold_loss, inter_intra_distance_metrics
from models.denoiser_fno1d import FNO1DDenoiser
from models.denoiser_rescnn_tcn import ResCNN1DDenoiser, TCN1DDenoiser
from models.denoiser_transformer1d import Transformer1DDenoiser
from models.denoiser_unet import DenoiserUNet1D
from models.denoiser_vit1d import PODViT1DDenoiser, ViT1DDenoiser
from models.frequency_encoder import FrequencyEncoder
from models.pod_denoisers import (
    ClassWisePODTrueWNO1DDenoiser,
    NoPODFrequencyAttnTrueWNO1DDenoiser,
    NoPODTrueWNO1DDenoiser,
    PODCoeffPredictor,
    PODDeepONetTimeDenoiser,
    PODFMEDNet,
    PODFNO1DDenoiser,
    PODFrequencyAttnTrueWNO1DDenoiser,
    PODResCNN1DDenoiser,
    PODTCN1DDenoiser,
    PODTransformer1DDenoiser,
    PODTrueWNO1DDenoiser,
    PODUWNO1DDenoiser,
    PODWNO1DDenoiser,
    UWNO1DDenoiser,
)
from pod_upper_bound_analysis import fit_pod
from transforms.frequency import batch_frequency_transform
from utils.metrics import diagnosis_metrics, frequency_metrics, separability_metrics, waveform_metrics
from utils.plotting import save_comparison_bar, save_confusion_matrix, save_fault_peak_preservation, save_latent_pca, save_latent_tsne, save_spectrum_examples, save_waveform_examples
from utils.save_utils import ensure_dir, load_config, print_config, save_json
from utils.seed import set_seed


def device_from(config):
    return torch.device("cuda" if config.get("device", "auto") == "auto" and torch.cuda.is_available() else config.get("device", "cpu"))


def make_noisy(clean, snrs, seed_offset=0):
    xs = clean.detach().cpu().numpy()
    out = []
    for i, x in enumerate(xs):
        snr = snrs[(i + seed_offset) % len(snrs)] if isinstance(snrs, list) else snrs
        out.append(add_awgn(x, snr, np.random.default_rng(90_000 + seed_offset + i)))
    return torch.from_numpy(np.stack(out)).to(clean.device).float()


def normalize_batch(x, mode="none", eps=1e-8):
    mode = (mode or "none").lower()
    if mode in {"none", "raw", ""}:
        meta = {"mode": "none"}
        return x, meta
    if mode in {"zscore", "zero", "standard"}:
        center = x.mean(dim=-1, keepdim=True)
        scale = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - center) / scale, {"mode": "affine", "center": center, "scale": scale}
    if mode in {"rms", "rmsnorm"}:
        center = torch.zeros_like(x[..., :1])
        scale = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True)).clamp_min(eps)
        return x / scale, {"mode": "affine", "center": center, "scale": scale}
    if mode in {"minmax", "min_max"}:
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        scale = (x_max - x_min).clamp_min(eps)
        return 2.0 * (x - x_min) / scale - 1.0, {"mode": "minmax", "min": x_min, "scale": scale}
    raise ValueError(f"Unknown sample_normalization mode: {mode}")


def denormalize_batch(x, meta):
    mode = meta.get("mode", "none")
    if mode == "none":
        return x
    if mode == "affine":
        return x * meta["scale"] + meta["center"]
    if mode == "minmax":
        return (x + 1.0) * 0.5 * meta["scale"] + meta["min"]
    raise ValueError(f"Unknown normalization metadata mode: {mode}")


def apply_normalization_meta(x, meta):
    mode = meta.get("mode", "none")
    if mode == "none":
        return x
    if mode == "affine":
        return (x - meta["center"]) / meta["scale"]
    if mode == "minmax":
        return 2.0 * (x - meta["min"]) / meta["scale"] - 1.0
    raise ValueError(f"Unknown normalization metadata mode: {mode}")


def make_noisy_for_model(clean_raw, snrs, config, seed_offset=0):
    mode = config["data"].get("sample_normalization", "none")
    noisy_raw = make_noisy(clean_raw, snrs, seed_offset)
    source = config["data"].get("normalization_source", "clean")
    if source not in {"clean", "noisy"}:
        raise ValueError(f"Unknown normalization source: {source}")
    # New protocols can use an input-only scale without changing archived runs.
    _, norm_meta = normalize_batch(noisy_raw if source == "noisy" else clean_raw, mode)
    clean_model = apply_normalization_meta(clean_raw, norm_meta)
    noisy_model = apply_normalization_meta(noisy_raw, norm_meta)
    return clean_model, noisy_model, noisy_raw, norm_meta


def load_reference_encoder(config, device):
    state = torch.load(config["paths"]["stage1_encoder"], map_location=device)
    enc = FrequencyEncoder(**config["reference_encoder_model"]).to(device)
    enc.load_state_dict(state["model"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.no_grad()
def collect_clean_arrays(config, split, device):
    loader = build_dataloader(config, split, shuffle=False)
    xs, specs, labels = [], [], []
    for batch in loader:
        clean = batch["signal"].to(device).float()
        spec = batch_frequency_transform(clean, **config["frequency"])
        xs.append(clean.cpu().numpy())
        specs.append(spec.squeeze(1).cpu().numpy())
        if "label" in batch:
            labels.extend(batch["label"].numpy().tolist())
    return np.concatenate(xs).astype(np.float32), np.concatenate(specs).astype(np.float32), np.asarray(labels) if labels else None


def ensure_frequency_pod(config, device):
    rank = int(config["pod"]["rank"])
    path = Path(config["paths"]["pod_analysis_dir"]) / f"pod_basis_rank_{rank}.npz"
    if path.exists():
        data = np.load(path)
        return {
            "mean": data["mean"].astype(np.float32),
            "basis": data["basis"].astype(np.float32),
            "rank": rank,
        }
    _, train_spec, _ = collect_clean_arrays(config, "train", device)
    mu, _, vt, _, energy = fit_pod(train_spec, config["pod"].get("standardize", False))
    ensure_dir(path.parent)
    np.savez(path, mean=mu.squeeze(0), basis=vt[:rank], rank=rank, energy_ratio=energy[rank - 1])
    return {"mean": mu.squeeze(0).astype(np.float32), "basis": vt[:rank].astype(np.float32), "rank": rank}


def ensure_classwise_frequency_pod(config, device):
    rank = int(config["pod"]["rank"])
    path = Path(config["paths"]["pod_analysis_dir"]) / f"classwise_pod_basis_rank_{rank}.npz"
    if path.exists():
        data = np.load(path)
        return {
            "mean": data["mean"].astype(np.float32),
            "basis": data["basis"].astype(np.float32),
            "rank": rank,
            "classes": data["classes"].astype(np.int64),
        }
    _, train_spec, labels = collect_clean_arrays(config, "train", device)
    if labels is None:
        raise ValueError("Class-wise POD requires labels in the training data.")
    classes = np.asarray(sorted(np.unique(labels)), dtype=np.int64)
    means, bases, energy_ratios = [], [], []
    for cls in classes:
        spec_c = train_spec[labels == cls]
        if len(spec_c) <= rank:
            raise ValueError(f"Class {cls} has only {len(spec_c)} samples, not enough for rank {rank}.")
        mu, _, vt, _, energy = fit_pod(spec_c, config["pod"].get("standardize", False))
        means.append(mu.squeeze(0).astype(np.float32))
        bases.append(vt[:rank].astype(np.float32))
        energy_ratios.append(float(energy[rank - 1]))
    ensure_dir(path.parent)
    np.savez(
        path,
        mean=np.stack(means),
        basis=np.stack(bases),
        classes=classes,
        rank=rank,
        energy_ratio=np.asarray(energy_ratios, dtype=np.float32),
    )
    return {"mean": np.stack(means), "basis": np.stack(bases), "rank": rank, "classes": classes}


def ensure_time_pod(config, device):
    rank = int(config["pod"]["rank"])
    path = Path(config["paths"]["pod_denoising_ablation_dir"]) / f"time_pod_basis_rank_{rank}.npz"
    if path.exists():
        data = np.load(path)
        return {"mean": data["mean"].astype(np.float32), "basis": data["basis"].astype(np.float32)}
    train_x, _, _ = collect_clean_arrays(config, "train", device)
    flat = train_x.squeeze(1)
    mu, _, vt, _, energy = fit_pod(flat, False)
    ensure_dir(path.parent)
    np.savez(path, mean=mu.squeeze(0), basis=vt[:rank], rank=rank, energy_ratio=energy[rank - 1])
    return {"mean": mu.squeeze(0).astype(np.float32), "basis": vt[:rank].astype(np.float32)}


class PODProjector(torch.nn.Module):
    def __init__(self, mean, basis):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1))
        self.register_buffer("basis", torch.as_tensor(basis, dtype=torch.float32))

    def coeff(self, spectrum):
        s = spectrum.squeeze(1)
        return (s - self.mean) @ self.basis.T

    def reconstruct(self, coeff):
        return (self.mean + coeff @ self.basis).unsqueeze(1)


class ClassWisePODProjector(torch.nn.Module):
    def __init__(self, mean, basis, classes):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("basis", torch.as_tensor(basis, dtype=torch.float32))
        self.register_buffer("classes", torch.as_tensor(classes, dtype=torch.long))

    def _label_index(self, labels):
        labels = labels.long()
        if torch.equal(self.classes, torch.arange(len(self.classes), device=self.classes.device)):
            return labels
        idx = torch.empty_like(labels)
        for i, cls in enumerate(self.classes):
            idx[labels == cls] = i
        return idx

    def coeff_all(self, spectrum):
        s = spectrum.squeeze(1)
        centered = s[:, None, :] - self.mean[None, :, :]
        return torch.einsum("bcd,crd->bcr", centered, self.basis)

    def coeff(self, spectrum, labels=None, class_prob=None):
        coeff_all = self.coeff_all(spectrum)
        if labels is not None:
            idx = self._label_index(labels.to(coeff_all.device))
            return coeff_all[torch.arange(coeff_all.shape[0], device=coeff_all.device), idx]
        if class_prob is not None:
            return torch.sum(class_prob.unsqueeze(-1) * coeff_all, dim=1)
        return torch.mean(coeff_all, dim=1)

    def reconstruct(self, coeff, class_prob=None, labels=None):
        if coeff.dim() == 3:
            recon_all = self.mean[None, :, :] + torch.einsum("bcr,crd->bcd", coeff, self.basis)
            if labels is not None:
                idx = self._label_index(labels.to(recon_all.device))
                return recon_all[torch.arange(recon_all.shape[0], device=recon_all.device), idx].unsqueeze(1)
            if class_prob is None:
                class_prob = torch.full(
                    (coeff.shape[0], coeff.shape[1]),
                    1.0 / coeff.shape[1],
                    device=coeff.device,
                    dtype=coeff.dtype,
                )
            return torch.sum(class_prob.unsqueeze(-1) * recon_all, dim=1).unsqueeze(1)
        if labels is not None:
            idx = self._label_index(labels.to(coeff.device))
            mean = self.mean[idx]
            basis = self.basis[idx]
            return (mean + torch.einsum("br,brd->bd", coeff, basis)).unsqueeze(1)
        raise ValueError("Class-wise POD reconstruction needs coeff [B,C,R] or labels for coeff [B,R].")


def is_classwise_pod_version(version):
    return version == "P27_classwise_pod_true_wno_direct"


def build_model(version, config, freq_pod, time_pod, device):
    rank = int(config["pod"]["rank"])
    base = int(config["model"]["base_channels"])
    residual_scale = config["model"]["residual_scale"] if config["model"].get("use_residual_limit", False) else 1.0
    if version == "P0_unet_v5":
        return DenoiserUNet1D(base_channels=base, mode="residual", residual_scale=1.0).to(device)
    if version == "P2_pod_coeff_only":
        return PODCoeffPredictor(rank=rank).to(device)
    if version == "P3_pod_deeponet_time":
        return PODDeepONetTimeDenoiser(time_pod["mean"], time_pod["basis"]).to(device)
    if version == "P10_pod_wno_gate":
        return PODWNO1DDenoiser(rank=rank, width=64, depth=4, levels=3, residual=True, residual_scale=residual_scale).to(device)
    if version == "P11_pod_true_wno_gate":
        return PODTrueWNO1DDenoiser(rank=rank, width=64, depth=4, levels=3, residual=True, residual_scale=residual_scale).to(device)
    if version == "P12_pod_true_wno_direct":
        return PODTrueWNO1DDenoiser(rank=rank, width=64, depth=4, levels=3, residual=False, residual_scale=residual_scale).to(device)
    if version == "P12_no_pod_true_wno_direct":
        return NoPODTrueWNO1DDenoiser(width=64, depth=4, levels=3, residual=False, residual_scale=residual_scale).to(device)
    if version == "P28_no_pod_true_wno_large_direct":
        return NoPODTrueWNO1DDenoiser(width=128, depth=5, levels=3, residual=False, residual_scale=residual_scale).to(device)
    if version == "P29_pod_true_wno_large_direct":
        return PODTrueWNO1DDenoiser(rank=rank, width=128, depth=5, levels=3, residual=False, residual_scale=residual_scale).to(device)
    if version == "P30_uwno_direct":
        return UWNO1DDenoiser(
            base_channels=base,
            width=128,
            depth=3,
            levels=3,
            residual=False,
            residual_scale=residual_scale,
        ).to(device)
    if version == "P31_pod_uwno_direct":
        return PODUWNO1DDenoiser(
            rank=rank,
            base_channels=base,
            width=128,
            depth=3,
            levels=3,
            residual=False,
            residual_scale=residual_scale,
        ).to(device)
    if version == "P27_classwise_pod_true_wno_direct":
        return ClassWisePODTrueWNO1DDenoiser(
            num_classes=len(freq_pod["classes"]),
            rank=rank,
            width=64,
            depth=4,
            levels=3,
            residual=False,
            residual_scale=residual_scale,
        ).to(device)
    if version == "P15_fno_direct_no_pod":
        return FNO1DDenoiser(width=64, modes=64, depth=4, residual=False, residual_scale=residual_scale).to(device)
    if version == "P16_pod_fno_direct":
        return PODFNO1DDenoiser(rank=rank, width=64, modes=64, depth=4, residual=False, residual_scale=residual_scale).to(device)
    if version == "P17_rescnn_no_pod":
        return ResCNN1DDenoiser(width=64, depth=8, kernel_size=7, residual=False, residual_scale=residual_scale).to(device)
    if version == "P18_tcn_no_pod":
        return TCN1DDenoiser(width=64, depth=8, kernel_size=5, residual=False, residual_scale=residual_scale).to(device)
    if version == "P19_pod_rescnn_direct":
        return PODResCNN1DDenoiser(rank=rank, width=64, depth=8, kernel_size=7, residual=False, residual_scale=residual_scale).to(device)
    if version == "P20_pod_tcn_direct":
        return PODTCN1DDenoiser(rank=rank, width=64, depth=8, kernel_size=5, residual=False, residual_scale=residual_scale).to(device)
    if version == "P21_transformer_no_pod":
        return Transformer1DDenoiser(width=128, depth=4, heads=4, patch_size=16, residual=False, residual_scale=residual_scale).to(device)
    if version == "P24_pod_transformer_direct":
        return PODTransformer1DDenoiser(rank=rank, width=128, depth=4, heads=4, patch_size=16, residual=False, residual_scale=residual_scale).to(device)
    if version == "P25_vit_no_pod":
        return ViT1DDenoiser(width=128, depth=4, heads=4, patch_size=16, residual=False, residual_scale=residual_scale).to(device)
    if version == "P26_pod_vit_direct":
        return PODViT1DDenoiser(rank=rank, width=128, depth=4, heads=4, patch_size=16, residual=False, residual_scale=residual_scale).to(device)
    if version == "P13_pod_fattn_true_wno_direct":
        return PODFrequencyAttnTrueWNO1DDenoiser(
            rank=rank,
            width=64,
            depth=4,
            levels=3,
            residual=False,
            residual_scale=residual_scale,
            max_modes=256,
            attn_every_block=True,
        ).to(device)
    if version == "P13_no_pod_fattn_true_wno_direct":
        return NoPODFrequencyAttnTrueWNO1DDenoiser(
            width=64,
            depth=4,
            levels=3,
            residual=False,
            residual_scale=residual_scale,
            max_modes=256,
            attn_every_block=True,
        ).to(device)
    if version == "P14_pod_fattn_bottleneck_true_wno_direct":
        return PODFrequencyAttnTrueWNO1DDenoiser(
            rank=rank,
            width=64,
            depth=4,
            levels=3,
            residual=False,
            residual_scale=residual_scale,
            max_modes=256,
            attn_every_block=False,
        ).to(device)
    if version == "P14_no_pod_fattn_bottleneck_true_wno_direct":
        return NoPODFrequencyAttnTrueWNO1DDenoiser(
            width=64,
            depth=4,
            levels=3,
            residual=False,
            residual_scale=residual_scale,
            max_modes=256,
            attn_every_block=False,
        ).to(device)
    fusion = {
        "P4_pod_fmed_bottleneck": "bottleneck",
        "P5_pod_fmed_film": "film",
        "P6_pod_fmed_gate": "gate",
        "P7_pod_fmed_no_pod_spec": "film",
        "P8_pod_fmed_no_pod_coeff": "film",
        "P9_pod_fmed_no_graph": "film",
    }[version]
    return PODFMEDNet(rank=rank, spectrum_len=len(freq_pod["mean"]), base_channels=base, fusion_mode=fusion, residual_scale=residual_scale).to(device)


def active_losses(version):
    if version == "P0_unet_v5":
        return {"rec", "freq", "out", "graph"}
    if version == "P2_pod_coeff_only":
        return {"pod_coeff", "pod_spec"}
    if version == "P3_pod_deeponet_time":
        return {"rec", "time_coeff", "freq", "out"}
    if version == "P10_pod_wno_gate":
        return {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph"}
    if version == "P11_pod_true_wno_gate":
        return {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph"}
    if version == "P12_pod_true_wno_direct":
        return {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph"}
    if version == "P12_no_pod_true_wno_direct":
        return {"rec", "freq", "out", "graph"}
    if version == "P28_no_pod_true_wno_large_direct":
        return {"rec", "freq", "out", "graph"}
    if version == "P29_pod_true_wno_large_direct":
        return {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph"}
    if version == "P30_uwno_direct":
        return {"rec", "freq", "out", "graph"}
    if version == "P31_pod_uwno_direct":
        return {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph"}
    if version == "P27_classwise_pod_true_wno_direct":
        return {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph", "cls"}
    if version == "P15_fno_direct_no_pod":
        return {"rec", "freq", "out", "graph"}
    if version == "P16_pod_fno_direct":
        return {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph"}
    if version == "P17_rescnn_no_pod":
        return {"rec", "freq", "out", "graph"}
    if version == "P18_tcn_no_pod":
        return {"rec", "freq", "out", "graph"}
    if version == "P19_pod_rescnn_direct":
        return {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph"}
    if version == "P20_pod_tcn_direct":
        return {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph"}
    if version == "P21_transformer_no_pod":
        return {"rec", "freq", "out", "graph"}
    if version == "P24_pod_transformer_direct":
        return {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph"}
    if version == "P25_vit_no_pod":
        return {"rec", "freq", "out", "graph"}
    if version == "P26_pod_vit_direct":
        return {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph"}
    if version == "P13_pod_fattn_true_wno_direct":
        return {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph"}
    if version == "P13_no_pod_fattn_true_wno_direct":
        return {"rec", "freq", "out", "graph"}
    if version == "P14_pod_fattn_bottleneck_true_wno_direct":
        return {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph"}
    if version == "P14_no_pod_fattn_bottleneck_true_wno_direct":
        return {"rec", "freq", "out", "graph"}
    base = {"rec", "freq", "pod_coeff", "pod_spec", "out", "graph"}
    if version == "P7_pod_fmed_no_pod_spec":
        base.remove("pod_spec")
    if version == "P8_pod_fmed_no_pod_coeff":
        base.remove("pod_coeff")
    if version == "P9_pod_fmed_no_graph":
        base.remove("graph")
    return base


def compute_terms(version, model, e_ref, pod, clean, noisy, config, labels=None):
    losses = active_losses(version)
    clean_raw = denormalize_batch(clean, noisy.get("norm_meta", {"mode": "none"})) if isinstance(noisy, dict) else clean
    noisy_model = noisy["signal"] if isinstance(noisy, dict) else noisy
    s_clean = batch_frequency_transform(clean_raw, **config["frequency"])
    with torch.no_grad():
        z_clean = e_ref(s_clean)
        a_true = pod.coeff(s_clean)
    x_hat, aux = model(noisy_model, return_aux=True) if version != "P0_unet_v5" else (model(noisy_model), {})
    zero = clean.new_tensor(0.0)
    terms = {k: zero for k in ["rec", "freq", "pod_coeff", "pod_spec", "out", "graph", "time_coeff", "cls"]}
    z_hat = z_clean
    s_hat = None
    if x_hat is not None:
        x_hat_raw = denormalize_batch(x_hat, noisy.get("norm_meta", {"mode": "none"})) if isinstance(noisy, dict) else x_hat
        s_hat = batch_frequency_transform(x_hat_raw, **config["frequency"])
        z_hat = e_ref(s_hat)
        terms["rec"] = F.l1_loss(x_hat, clean) if "rec" in losses else zero
        terms["freq"] = F.l1_loss(s_hat, s_clean) if "freq" in losses else zero
        terms["out"] = F.mse_loss(z_hat, z_clean) if "out" in losses else zero
        terms["graph"] = graph_manifold_loss(z_hat, z_clean, k=10) if "graph" in losses else zero
    a_hat = aux.get("a_hat")
    if aux.get("a_hat_all") is not None:
        if labels is None:
            raise ValueError("Class-wise POD loss requires labels.")
        labels = labels.to(clean.device).long()
        a_true_cls = pod.coeff(s_clean, labels=labels)
        a_hat_cls = aux["a_hat_all"][torch.arange(labels.shape[0], device=labels.device), labels]
        s_pod_hat = pod.reconstruct(aux["a_hat_all"], class_prob=aux.get("class_prob"))
        terms["pod_coeff"] = F.mse_loss(a_hat_cls, a_true_cls) if "pod_coeff" in losses else zero
        terms["pod_spec"] = F.l1_loss(s_pod_hat, s_clean) if "pod_spec" in losses else zero
        terms["cls"] = F.cross_entropy(aux["class_logits"], labels) if "cls" in losses else zero
        aux["s_pod_hat"] = s_pod_hat
    elif a_hat is not None:
        s_pod_hat = pod.reconstruct(a_hat)
        terms["pod_coeff"] = F.mse_loss(a_hat, a_true) if "pod_coeff" in losses else zero
        terms["pod_spec"] = F.l1_loss(s_pod_hat, s_clean) if "pod_spec" in losses else zero
        aux["s_pod_hat"] = s_pod_hat
    w = config["loss_weights"]
    terms["total"] = (
        w["lambda_rec"] * terms["rec"]
        + w["lambda_freq"] * terms["freq"]
        + w["lambda_pod_coeff"] * terms["pod_coeff"]
        + w["lambda_pod_spec"] * terms["pod_spec"]
        + w["lambda_out"] * terms["out"]
        + w["lambda_graph"] * terms["graph"]
        + w.get("lambda_cls", 0.05) * terms["cls"]
        + 0.01 * terms["time_coeff"]
    )
    return x_hat, aux, z_hat, z_clean, terms


@torch.no_grad()
def collect_pack(version, config, model, e_ref, pod, loader, device, snr=-5):
    clean_l, noisy_l, den_l, sc_l, sn_l, sd_l, sp_l, zc_l, zd_l, labels = [], [], [], [], [], [], [], [], [], []
    coeff_mse_l = []
    selector_correct, selector_total = 0, 0
    for bi, batch in enumerate(loader):
        clean_raw = batch["signal"].to(device).float()
        if snr is None:
            clean_model, norm_meta = normalize_batch(clean_raw, config["data"].get("sample_normalization", "none"))
            noisy_model = clean_model
            noisy_raw = clean_raw
        else:
            clean_model, noisy_model, noisy_raw, norm_meta = make_noisy_for_model(clean_raw, snr, config, bi * 1000)
        s_clean = batch_frequency_transform(clean_raw, **config["frequency"])
        s_noisy = batch_frequency_transform(noisy_raw, **config["frequency"])
        labels_t = batch["label"].to(device).long() if "label" in batch else None
        if model is None:
            x_hat_model, aux = noisy_model, {}
        elif version == "P0_unet_v5":
            x_hat_model, aux = model(noisy_model), {}
        else:
            x_hat_model, aux = model(noisy_model, return_aux=True)
        if x_hat_model is None:
            x_hat_model = noisy_model
        x_hat_raw = denormalize_batch(x_hat_model, norm_meta)
        s_den = batch_frequency_transform(x_hat_raw, **config["frequency"])
        zc, zd = e_ref(s_clean), e_ref(s_den)
        a_hat = aux.get("a_hat")
        if aux.get("a_hat_all") is not None:
            if labels_t is None:
                raise ValueError("Class-wise POD evaluation requires labels.")
            a_true = pod.coeff(s_clean, labels=labels_t)
            label_idx = pod._label_index(labels_t)
            a_hat_cls = aux["a_hat_all"][torch.arange(labels_t.shape[0], device=device), label_idx]
            s_pod = pod.reconstruct(aux["a_hat_all"], class_prob=aux.get("class_prob"))
            coeff_mse_l.append(F.mse_loss(a_hat_cls, a_true, reduction="none").mean(dim=1).cpu().numpy())
            pred_idx = torch.argmax(aux["class_logits"], dim=1)
            selector_correct += int((pred_idx == label_idx).sum().item())
            selector_total += int(labels_t.numel())
        elif a_hat is not None:
            a_true = pod.coeff(s_clean)
            s_pod = pod.reconstruct(a_hat)
            coeff_mse_l.append(F.mse_loss(a_hat, a_true, reduction="none").mean(dim=1).cpu().numpy())
        else:
            s_pod = s_den
        clean_l.append(clean_raw.cpu().numpy()); noisy_l.append(noisy_raw.cpu().numpy()); den_l.append(x_hat_raw.cpu().numpy())
        sc_l.append(s_clean.cpu().numpy()); sn_l.append(s_noisy.cpu().numpy()); sd_l.append(s_den.cpu().numpy()); sp_l.append(s_pod.cpu().numpy())
        zc_l.append(zc.cpu().numpy()); zd_l.append(zd.cpu().numpy())
        if "label" in batch:
            labels.extend(batch["label"].numpy().tolist())
    return {
        "clean": np.concatenate(clean_l), "noisy": np.concatenate(noisy_l), "denoised": np.concatenate(den_l),
        "s_clean": np.concatenate(sc_l), "s_noisy": np.concatenate(sn_l), "s_denoised": np.concatenate(sd_l), "s_pod": np.concatenate(sp_l),
        "z_clean": np.concatenate(zc_l), "z_denoised": np.concatenate(zd_l),
        "labels": np.asarray(labels) if labels else None,
        "pod_coeff_mse": float(np.mean(np.concatenate(coeff_mse_l))) if coeff_mse_l else np.nan,
        "class_selector_accuracy": float(selector_correct / selector_total) if selector_total else np.nan,
    }


def evaluate_version(version, config, model, e_ref, pod, train_pack, device, out_dir):
    test_loader = build_dataloader(config, "test", shuffle=False)
    pack = collect_pack(version, config, model, e_ref, pod, test_loader, device, snr=-5)
    row = {"version": version, **waveform_metrics(pack["clean"], pack["denoised"], pack["noisy"]), **frequency_metrics(pack["s_clean"], pack["s_denoised"])}
    pod_freq = frequency_metrics(pack["s_clean"], pack["s_pod"])
    row.update({"pod_spectral_rmse": pod_freq["spectral_rmse"], "pod_envelope_corr": pod_freq["envelope_corr"], "pod_coeff_mse": pack["pod_coeff_mse"]})
    if not np.isnan(pack.get("class_selector_accuracy", np.nan)):
        row["class_selector_accuracy"] = pack["class_selector_accuracy"]
    row["latent_mse"] = float(np.mean((pack["z_denoised"] - pack["z_clean"]) ** 2))
    if pack["labels"] is not None:
        row.update(inter_intra_distance_metrics(pack["z_denoised"], pack["labels"]))
        sep = separability_metrics(pack["z_denoised"], pack["labels"])
        row.update({"silhouette": sep.get("silhouette", np.nan), "davies_bouldin": sep.get("davies_bouldin", np.nan), "calinski_harabasz": sep.get("calinski_harabasz", np.nan)})
        diag, cm = diagnosis_metrics(train_pack["z_clean"], train_pack["labels"], pack["z_denoised"], pack["labels"])
        row.update(diag)
        save_confusion_matrix(cm, out_dir / "confusion_matrix.png")
    save_waveform_examples(pack["clean"], pack["noisy"], pack["denoised"], pack["labels"], out_dir / "waveform_examples.png")
    save_spectrum_examples(pack["s_clean"], pack["s_pod"], pack["s_denoised"], pack["labels"], out_dir / "envelope_spectrum_examples.png")
    peak_rows = save_fault_peak_preservation(
        pack["s_clean"],
        pack["s_noisy"],
        pack["s_denoised"],
        pack["labels"],
        out_dir / "fault_peak_preservation.png",
        out_dir / "fault_peak_preservation_metrics.csv",
    )
    if peak_rows:
        row["fault_peak_noisy_rel_error"] = float(np.mean([r["noisy_relative_abs_error"] for r in peak_rows]))
        row["fault_peak_denoised_rel_error"] = float(np.mean([r["denoised_relative_abs_error"] for r in peak_rows]))
        row["fault_peak_error_reduction"] = row["fault_peak_noisy_rel_error"] - row["fault_peak_denoised_rel_error"]
        row["fault_peak_noisy_peak_to_floor"] = float(np.mean([r["noisy_peak_to_floor"] for r in peak_rows]))
        row["fault_peak_denoised_peak_to_floor"] = float(np.mean([r["denoised_peak_to_floor"] for r in peak_rows]))
    combo = np.concatenate([pack["z_clean"], pack["z_denoised"]])
    combo_l = np.array(["clean"] * len(pack["z_clean"]) + ["denoised"] * len(pack["z_denoised"]))
    save_latent_pca(combo, combo_l, out_dir / "latent_pca_clean_denoised.png", title=f"{version} latent PCA")
    save_latent_tsne(combo, combo_l, out_dir / "latent_tsne_clean_denoised.png", title=f"{version} latent t-SNE")
    by_snr = []
    for snr in config["data"]["test_snr_list"]:
        p = collect_pack(version, config, model, e_ref, pod, test_loader, device, snr=snr)
        by_snr.append({"version": version, "snr_db": snr, **waveform_metrics(p["clean"], p["denoised"], p["noisy"]), **frequency_metrics(p["s_clean"], p["s_denoised"])})
    return row, by_snr


def write_csv(path, rows):
    path = Path(path); ensure_dir(path.parent)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"Saved: {path.resolve()}")


def inference_time_ms(model, sample, version, repeat=50):
    model.eval()
    device = sample.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            if version == "P0_unet_v5":
                model(sample)
            else:
                model(sample, return_aux=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--versions", nargs="*", default=None)
    args = ap.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    print_config(config)
    set_seed(config["seed"])
    device = device_from(config)
    base = ensure_dir(config["paths"]["pod_denoising_ablation_dir"])
    freq_pod = ensure_frequency_pod(config, device)
    class_freq_pod = ensure_classwise_frequency_pod(config, device)
    time_pod = ensure_time_pod(config, device)
    global_pod = PODProjector(freq_pod["mean"], freq_pod["basis"]).to(device)
    classwise_pod = ClassWisePODProjector(class_freq_pod["mean"], class_freq_pod["basis"], class_freq_pod["classes"]).to(device)
    e_ref = load_reference_encoder(config, device)
    train_loader = build_dataloader(config, "train", shuffle=False)
    val_loader = build_dataloader(config, "val", shuffle=False)
    train_pack = collect_pack("clean", config, None, e_ref, global_pod, train_loader, device, snr=None)
    versions = args.versions or [
        "P0_unet_v5",
        "P2_pod_coeff_only",
        "P3_pod_deeponet_time",
        "P4_pod_fmed_bottleneck",
        "P5_pod_fmed_film",
        "P6_pod_fmed_gate",
        "P7_pod_fmed_no_pod_spec",
        "P8_pod_fmed_no_pod_coeff",
        "P9_pod_fmed_no_graph",
    ]
    summary, by_snr_all = [], []
    for version in versions:
        out = ensure_dir(base / version)
        save_json(config, out / "config.json")
        version_freq_pod = class_freq_pod if is_classwise_pod_version(version) else freq_pod
        version_pod = classwise_pod if is_classwise_pod_version(version) else global_pod
        model = build_model(version, config, version_freq_pod, time_pod, device)
        opt = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
        train_rows, val_rows = [], []
        best = {"score": -1e18}
        for epoch in range(1, config["training"]["epochs"] + 1):
            model.train()
            sums = {k: 0.0 for k in ["total", "rec", "freq", "pod_coeff", "pod_spec", "out", "graph", "cls"]}
            steps = 0
            for bi, batch in enumerate(train_loader):
                clean_raw = batch["signal"].to(device).float()
                clean_model, noisy_model, _, norm_meta = make_noisy_for_model(clean_raw, config["data"]["train_snr_choices"], config, bi * 1000)
                noisy = {"signal": noisy_model, "norm_meta": norm_meta}
                labels = batch["label"].to(device).long() if "label" in batch else None
                _, _, _, _, terms = compute_terms(version, model, e_ref, version_pod, clean_model, noisy, config, labels=labels)
                opt.zero_grad(set_to_none=True); terms["total"].backward()
                if config["training"].get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
                opt.step()
                for k in sums:
                    sums[k] += float(terms[k].detach().cpu())
                steps += 1
            tr = {"epoch": epoch, **{f"train_{k}": v / max(1, steps) for k, v in sums.items()}}
            train_rows.append(tr)
            if epoch % config["training"]["eval_every"] == 0:
                val_pack = collect_pack(version, config, model, e_ref, version_pod, val_loader, device, snr=-5)
                vm = {**waveform_metrics(val_pack["clean"], val_pack["denoised"], val_pack["noisy"]), **frequency_metrics(val_pack["s_clean"], val_pack["s_denoised"])}
                vm["latent_mse"] = float(np.mean((val_pack["z_denoised"] - val_pack["z_clean"]) ** 2))
                score = vm["snr_improvement"] + 0.5 * vm["envelope_corr"] - 0.5 * vm["rmse"] - 0.1 * vm["latent_mse"]
                val_rows.append({"epoch": epoch, "score": score, **vm})
                if score > best["score"]:
                    best["score"] = score
                    torch.save({"model": model.state_dict(), "config": config, "version": version, "epoch": epoch}, out / "best_by_score.pth")
                print(f"{version} epoch {epoch}: train={tr['train_total']:.5f} snr={vm['snr_improvement']:.3f} rmse={vm['rmse']:.4f} env={vm['envelope_corr']:.3f} latent={vm['latent_mse']:.4f}")
        torch.save({"model": model.state_dict(), "config": config, "version": version}, out / "last.pth")
        write_csv(out / "train_log.csv", train_rows)
        write_csv(out / "val_log.csv", val_rows)
        if (out / "best_by_score.pth").exists():
            state = torch.load(out / "best_by_score.pth", map_location=device)
            model.load_state_dict(state["model"])
        row, by_snr = evaluate_version(version, config, model, e_ref, version_pod, train_pack, device, out)
        sample = next(iter(val_loader))["signal"].to(device).float()[:16]
        row["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
        row["inference_time_ms"] = inference_time_ms(model, sample, version)
        summary.append(row); by_snr_all.extend(by_snr)
        write_csv(out / "pod_denoising_metrics.csv", [row])
    write_csv(base / "pod_denoising_summary.csv", summary)
    write_csv(base / "pod_denoising_by_snr.csv", by_snr_all)
    save_comparison_bar(base / "pod_denoising_summary.csv", base / "pod_denoising_bar.png")
    best_snr = max(summary, key=lambda r: r["snr_improvement"])
    best_log = max(summary, key=lambda r: r.get("logistic_accuracy", -1))
    best_lat = min(summary, key=lambda r: r["latent_mse"])
    lines = [
        f"Best SNR: {best_snr['version']} ({best_snr['snr_improvement']:.4f})",
        f"Best Logistic: {best_log['version']} ({best_log.get('logistic_accuracy', np.nan):.4f})",
        f"Best latent_mse: {best_lat['version']} ({best_lat['latent_mse']:.6f})",
    ]
    (base / "pod_denoising_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {(base / 'pod_denoising_summary.txt').resolve()}")


if __name__ == "__main__":
    main()
