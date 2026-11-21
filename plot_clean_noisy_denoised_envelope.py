import argparse
from pathlib import Path

import numpy as np


def _as_2d_signals(x, name):
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    elif arr.ndim == 2:
        pass
    elif arr.ndim == 1:
        arr = arr[None, :]
    else:
        raise ValueError(f"{name} must have shape [N,L], [N,1,L], or [L], got {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr


def _hilbert_envelope_numpy(x):
    n = len(x)
    xf = np.fft.fft(x)
    h = np.zeros(n, dtype=np.float32)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    analytic = np.fft.ifft(xf * h)
    return np.abs(analytic)


def _fft_bandpass(sig, fs, bandpass):
    if bandpass is None:
        return sig
    low, high = float(bandpass[0]), float(bandpass[1])
    if not (0 <= low < high < fs / 2):
        raise ValueError(f"Invalid bandpass {bandpass}; expected 0 <= low < high < fs/2.")
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / float(fs))
    spec = np.fft.rfft(sig)
    mask = (freqs >= low) & (freqs <= high)
    return np.fft.irfft(spec * mask, n=len(sig))


def envelope_spectrum(x, fs, n_fft=8192, remove_mean=True, use_window=True, bandpass=None):
    """
    Compute Hilbert envelope spectrum.
    Return frequency axis and single-sided amplitude spectrum.

    Note: zero-padding via n_fft can make the displayed curve smoother, but it
    does not change the real frequency resolution, which is fs / len(x).
    """
    sig = np.asarray(x, dtype=np.float64).squeeze()
    if sig.ndim != 1:
        raise ValueError(f"Expected one-dimensional signal, got shape {sig.shape}.")
    if not np.isfinite(sig).all():
        raise ValueError("Signal contains NaN or Inf.")
    sig = sig - np.mean(sig)
    sig = _fft_bandpass(sig, fs, bandpass)
    envelope = _hilbert_envelope_numpy(sig)
    if remove_mean:
        envelope = envelope - np.mean(envelope)
    if use_window:
        envelope = envelope * np.hanning(len(envelope))
    spec = np.fft.rfft(envelope, n=int(n_fft))
    amp = np.abs(spec) * 2.0 / max(1, len(envelope))
    freqs = np.fft.rfftfreq(int(n_fft), d=1.0 / float(fs))
    return freqs, amp


def _mark_fault_lines(ax, fault_freq, max_freq):
    if fault_freq is None or float(fault_freq) <= 0:
        return
    base = float(fault_freq)
    harmonic = 1
    while harmonic * base <= max_freq:
        ax.axvline(harmonic * base, color="crimson", linestyle="--", linewidth=0.8, alpha=0.45)
        harmonic += 1


def plot_clean_noisy_denoised(
    clean,
    noisy,
    denoised,
    fs=25600,
    max_freq=1000,
    fault_freq=None,
    save_path=None,
    title_suffix=None,
    n_fft=8192,
    bandpass=None,
):
    """
    Plot 2x3 figure:
    row 1: time-domain waveform
    row 2: envelope spectrum
    columns: clean, noisy, denoised
    """
    import matplotlib.pyplot as plt

    clean = np.asarray(clean, dtype=np.float64).squeeze()
    noisy = np.asarray(noisy, dtype=np.float64).squeeze()
    denoised = np.asarray(denoised, dtype=np.float64).squeeze()
    if clean.ndim != 1 or noisy.ndim != 1 or denoised.ndim != 1:
        raise ValueError("clean, noisy, and denoised must be one-dimensional after squeeze.")
    if not (len(clean) == len(noisy) == len(denoised)):
        raise ValueError(f"Signal lengths differ: clean={len(clean)}, noisy={len(noisy)}, denoised={len(denoised)}.")
    for name, sig in [("clean", clean), ("noisy", noisy), ("denoised", denoised)]:
        if not np.isfinite(sig).all():
            raise ValueError(f"{name} contains NaN or Inf.")

    n = len(clean)
    time = np.arange(n) / float(fs)
    df_real = float(fs) / n
    print("Signal length:", n)
    print("Sampling frequency:", fs, "Hz")
    print("Real frequency resolution:", df_real, "Hz")

    series = [("Clean", clean, "#1f77b4"), ("Noisy", noisy, "#ff7f0e"), ("Denoised", denoised, "#2ca02c")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    y_min = min(float(np.min(s)) for _, s, _ in series)
    y_max = max(float(np.max(s)) for _, s, _ in series)
    pad = 0.05 * max(1e-12, y_max - y_min)

    spec_rows = []
    spec_max = 0.0
    for _, sig, _ in series:
        freqs, amp = envelope_spectrum(sig, fs, n_fft=n_fft, bandpass=bandpass)
        keep = freqs <= float(max_freq)
        spec_rows.append((freqs, amp, keep))
        if np.any(keep):
            spec_max = max(spec_max, float(np.max(amp[keep])))

    for col, (name, sig, color) in enumerate(series):
        ax = axes[0, col]
        ax.plot(time, sig, color=color, linewidth=0.8)
        ax.set_title(name if title_suffix is None else f"{name} {title_suffix}", fontsize=12)
        ax.set_xlabel("Time (s)", fontsize=10)
        if col == 0:
            ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.tick_params(labelsize=9)

        freqs, amp, keep = spec_rows[col]
        ax = axes[1, col]
        ax.plot(freqs[keep], amp[keep], color=color, linewidth=0.9)
        _mark_fault_lines(ax, fault_freq, max_freq)
        ax.set_xlim(0, float(max_freq))
        ax.set_ylim(0, spec_max * 1.05 if spec_max > 0 else 1.0)
        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        if col == 0:
            ax.set_ylabel("Envelope spectrum", fontsize=10)
        ax.tick_params(labelsize=9)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, facecolor="white")
        print("Saved figure:", save_path.resolve())
    plt.close(fig)


def _load_arrays(path):
    path = Path(path)
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = set(obj.files)
        clean_key = next((k for k in ["clean", "clean_array", "x_clean", "x", "signals"] if k in keys), None)
        noisy_key = next((k for k in ["noisy", "noisy_array", "y_noisy"] if k in keys), None)
        denoised_key = next((k for k in ["denoised", "denoised_array", "x_hat", "pred", "output"] if k in keys), None)
        label_key = next((k for k in ["labels", "label", "y"] if k in keys), None)
        if clean_key is None or noisy_key is None or denoised_key is None:
            raise ValueError(
                f"Could not find clean/noisy/denoised in {path}. Available keys={sorted(keys)}. "
                "Modify _load_arrays() key candidates if your field names differ."
            )
        labels = obj[label_key] if label_key else None
        return obj[clean_key], obj[noisy_key], obj[denoised_key], labels
    if obj.shape == () and isinstance(obj.item(), dict):
        data = obj.item()
        clean = data.get("clean") or data.get("x_clean") or data.get("x")
        noisy = data.get("noisy") or data.get("y_noisy")
        denoised = data.get("denoised") or data.get("x_hat") or data.get("pred")
        labels = data.get("labels") or data.get("label") or data.get("y")
        if clean is None or noisy is None or denoised is None:
            raise ValueError("Numpy dict must contain clean/noisy/denoised. Modify _load_arrays() if needed.")
        return clean, noisy, denoised, labels
    raise ValueError("A plain .npy array is ambiguous. Use .npz with clean/noisy/denoised keys or a dict .npy.")


def _make_noisy(clean, snr_db, seed=20260626):
    rng = np.random.default_rng(seed)
    x = np.asarray(clean, dtype=np.float32)
    power = np.mean(np.square(x), axis=-1, keepdims=True)
    noise_power = power / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32)
    return x + noise * np.sqrt(np.maximum(noise_power, 1e-12))


def _normalize_for_plot(clean, noisy, denoised, mode, eps=1e-8):
    mode = (mode or "none").lower()
    if mode in {"none", "raw", ""}:
        return clean, noisy, denoised
    if mode in {"rms", "rmsnorm"}:
        scale = np.sqrt(np.mean(np.square(clean), axis=-1, keepdims=True))
        scale = np.maximum(scale, eps)
        return clean / scale, noisy / scale, denoised / scale
    if mode in {"zscore", "zero", "standard"}:
        center = np.mean(clean, axis=-1, keepdims=True)
        scale = np.std(clean, axis=-1, keepdims=True)
        scale = np.maximum(scale, eps)
        return (clean - center) / scale, (noisy - center) / scale, (denoised - center) / scale
    if mode in {"minmax", "min_max"}:
        x_min = np.min(clean, axis=-1, keepdims=True)
        scale = np.max(clean, axis=-1, keepdims=True) - x_min
        scale = np.maximum(scale, eps)
        return (
            2.0 * (clean - x_min) / scale - 1.0,
            2.0 * (noisy - x_min) / scale - 1.0,
            2.0 * (denoised - x_min) / scale - 1.0,
        )
    raise ValueError(f"Unknown --plot-normalization mode: {mode}")


def _generate_npz_from_model(args):
    import torch

    from data.dataset import build_dataloader
    from run_pod_denoising_experiments import (
        build_model,
        denormalize_batch,
        device_from,
        ensure_frequency_pod,
        ensure_time_pod,
        make_noisy_for_model,
    )
    from utils.save_utils import load_config
    from utils.seed import set_seed

    config = load_config(args.config)
    set_seed(config["seed"])
    device = device_from(config)
    freq_pod = ensure_frequency_pod(config, device)
    time_pod = ensure_time_pod(config, device)
    model = build_model(args.version, config, freq_pod, time_pod, device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    clean_list, noisy_list, denoised_list, label_list = [], [], [], []
    loader = build_dataloader(config, "test", shuffle=False)
    need = max(args.max_samples, 1)
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            clean_raw = batch["signal"].to(device).float()
            _, noisy_model, noisy_raw, norm_meta = make_noisy_for_model(clean_raw, args.snr_db, config, seed_offset=batch_id)
            denoised_model = model(noisy_model, return_aux=False)
            denoised_raw = denormalize_batch(denoised_model, norm_meta)
            clean_list.append(clean_raw.detach().cpu().numpy())
            noisy_list.append(noisy_raw.detach().cpu().numpy())
            denoised_list.append(denoised_raw.detach().cpu().numpy())
            if "label" in batch:
                label_list.append(batch["label"].numpy())
            if sum(len(a) for a in clean_list) >= need:
                break

    clean = np.concatenate(clean_list, axis=0)[:need]
    noisy = np.concatenate(noisy_list, axis=0)[:need]
    denoised = np.concatenate(denoised_list, axis=0)[:need]
    labels = np.concatenate(label_list, axis=0)[:need] if label_list else None
    out = Path(args.input)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, clean=clean, noisy=noisy, denoised=denoised, labels=labels)
    print("Saved visualization arrays:", out.resolve())


def _select_indices(labels, sample_indices, per_class):
    n = len(labels) if labels is not None else None
    if sample_indices:
        return [int(i) for i in sample_indices]
    if labels is not None and per_class:
        labels = np.asarray(labels)
        return [int(np.where(labels == c)[0][0]) for c in np.unique(labels) if len(np.where(labels == c)[0])]
    count = 3 if n is None else min(3, n)
    return list(range(count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs_4096/pod_denoising_ablation/P12_pod_true_wno_direct/visualization_arrays.npz")
    parser.add_argument("--fig-dir", default="figures")
    parser.add_argument("--fs", type=float, default=25600)
    parser.add_argument("--max-freq", type=float, default=1000)
    parser.add_argument("--n-fft", type=int, default=8192)
    parser.add_argument("--fault-freq", type=float, default=None)
    parser.add_argument("--bandpass", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--sample-indices", type=int, nargs="*", default=None)
    parser.add_argument("--per-class", action="store_true", help="Draw one sample for each label.")
    parser.add_argument("--generate-from-model", action="store_true")
    parser.add_argument("--config", default="configs/pod_fmed_config_4096.py")
    parser.add_argument("--checkpoint", default="outputs_4096/pod_denoising_ablation/P12_pod_true_wno_direct/best_by_score.pth")
    parser.add_argument("--version", default="P12_pod_true_wno_direct")
    parser.add_argument("--snr-db", type=float, default=-6)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument(
        "--plot-normalization",
        default="none",
        choices=["none", "rms", "zscore", "minmax"],
        help="Normalize clean/noisy/denoised only for plotting, using clean sample statistics.",
    )
    args = parser.parse_args()

    if args.generate_from_model:
        _generate_npz_from_model(args)

    clean, noisy, denoised, labels = _load_arrays(args.input)
    clean = _as_2d_signals(clean, "clean")
    noisy = _as_2d_signals(noisy, "noisy")
    denoised = _as_2d_signals(denoised, "denoised")
    if not (clean.shape == noisy.shape == denoised.shape):
        raise ValueError(f"Array shapes differ: clean={clean.shape}, noisy={noisy.shape}, denoised={denoised.shape}.")
    clean, noisy, denoised = _normalize_for_plot(clean, noisy, denoised, args.plot_normalization)
    if args.plot_normalization != "none":
        print("Plot normalization:", args.plot_normalization)
    labels = np.asarray(labels).reshape(-1) if labels is not None else None
    if labels is not None and len(labels) != len(clean):
        raise ValueError(f"labels length {len(labels)} does not match signals length {len(clean)}.")

    print("Loaded samples:", len(clean))
    print("Signal length:", clean.shape[-1])
    print("Sampling frequency:", args.fs, "Hz")
    print("Real frequency resolution:", args.fs / clean.shape[-1], "Hz")

    indices = _select_indices(labels, args.sample_indices, args.per_class)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        suffix = None
        if labels is not None:
            suffix = f"(sample {idx}, label {int(labels[idx])})"
        else:
            suffix = f"(sample {idx})"
        save_path = fig_dir / f"clean_noisy_denoised_sample_{idx}.png"
        plot_clean_noisy_denoised(
            clean[idx],
            noisy[idx],
            denoised[idx],
            fs=args.fs,
            max_freq=args.max_freq,
            fault_freq=args.fault_freq,
            save_path=save_path,
            title_suffix=suffix,
            n_fft=args.n_fft,
            bandpass=args.bandpass,
        )


if __name__ == "__main__":
    main()
