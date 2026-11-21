from pathlib import Path

import numpy as np

from .save_utils import ensure_dir


def save_latent_pca(features, labels, path, title="latent PCA"):
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except Exception as exc:
        print(f"Skip PCA plot because dependency is unavailable: {exc}")
        return
    features = np.asarray(features)
    if len(features) < 2:
        return
    z = PCA(n_components=2).fit_transform(features)
    path = Path(path)
    ensure_dir(path.parent)
    plt.figure(figsize=(6, 5))
    if labels is None:
        plt.scatter(z[:, 0], z[:, 1], s=14)
    else:
        labels_arr = np.asarray(labels)
        if labels_arr.dtype.kind in {"U", "S", "O"}:
            _, labels_arr = np.unique(labels_arr, return_inverse=True)
        plt.scatter(z[:, 0], z[:, 1], c=labels_arr, s=14, cmap="tab10")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path.resolve()}")


def save_latent_tsne(features, labels, path, perplexity=30, title="latent t-SNE"):
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except Exception as exc:
        print(f"Skip t-SNE plot because dependency is unavailable: {exc}")
        return
    features = np.asarray(features)
    if len(features) < 8:
        return
    safe_perplexity = min(perplexity, max(2, len(features) // 3))
    z = TSNE(n_components=2, perplexity=safe_perplexity, init="pca", learning_rate="auto").fit_transform(features)
    path = Path(path)
    ensure_dir(path.parent)
    plt.figure(figsize=(6, 5))
    if labels is None:
        plt.scatter(z[:, 0], z[:, 1], s=14)
    else:
        labels_arr = np.asarray(labels)
        if labels_arr.dtype.kind in {"U", "S", "O"}:
            _, labels_arr = np.unique(labels_arr, return_inverse=True)
        plt.scatter(z[:, 0], z[:, 1], c=labels_arr, s=14, cmap="tab10")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path.resolve()}")


def save_waveform_examples(clean, noisy, denoised, labels, path, max_classes=9):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip waveform plot: {exc}")
        return
    clean = np.asarray(clean).squeeze(1)
    noisy = np.asarray(noisy).squeeze(1)
    denoised = np.asarray(denoised).squeeze(1)
    labels = np.asarray(labels) if labels is not None else np.zeros(len(clean), dtype=int)
    classes = np.unique(labels)[:max_classes]
    path = Path(path)
    ensure_dir(path.parent)
    fig, axes = plt.subplots(len(classes), 1, figsize=(10, max(2, 1.6 * len(classes))), squeeze=False)
    for ax, c in zip(axes[:, 0], classes):
        i = np.where(labels == c)[0][0]
        ax.plot(clean[i], label="clean", linewidth=1)
        ax.plot(noisy[i], label="noisy", linewidth=0.8, alpha=0.7)
        ax.plot(denoised[i], label="denoised", linewidth=0.8)
        ax.set_title(f"class {c}")
    axes[0, 0].legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path.resolve()}")


def save_confusion_matrix(cm, path, title="confusion matrix"):
    if cm is None:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip confusion matrix plot: {exc}")
        return
    path = Path(path)
    ensure_dir(path.parent)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Saved: {path.resolve()}")


def save_spectrum_examples(clean_spec, noisy_spec, denoised_spec, labels, path, max_classes=9):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip spectrum plot: {exc}")
        return
    clean_spec = np.asarray(clean_spec).squeeze(1)
    noisy_spec = np.asarray(noisy_spec).squeeze(1)
    denoised_spec = np.asarray(denoised_spec).squeeze(1)
    labels = np.asarray(labels) if labels is not None else np.zeros(len(clean_spec), dtype=int)
    classes = np.unique(labels)[:max_classes]
    path = Path(path)
    ensure_dir(path.parent)
    fig, axes = plt.subplots(len(classes), 1, figsize=(10, max(2, 1.6 * len(classes))), squeeze=False)
    for ax, c in zip(axes[:, 0], classes):
        i = np.where(labels == c)[0][0]
        ax.plot(clean_spec[i], label="clean", linewidth=1)
        ax.plot(noisy_spec[i], label="noisy", linewidth=0.8, alpha=0.7)
        ax.plot(denoised_spec[i], label="denoised", linewidth=0.8)
        ax.set_title(f"class {c}")
    axes[0, 0].legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path.resolve()}")


def _top_clean_peaks(spec, top_k=5, min_distance=12, ignore_bins=3):
    spec = np.asarray(spec, dtype=float)
    if spec.size <= 2 * ignore_bins + 3:
        return np.array([], dtype=int)
    candidates = []
    for i in range(max(1, ignore_bins), spec.size - 1):
        if spec[i] > spec[i - 1] and spec[i] >= spec[i + 1]:
            candidates.append(i)
    candidates = sorted(candidates, key=lambda idx: spec[idx], reverse=True)
    selected = []
    for idx in candidates:
        if all(abs(idx - j) >= min_distance for j in selected):
            selected.append(idx)
        if len(selected) >= top_k:
            break
    return np.array(sorted(selected), dtype=int)


def _local_noise_floor(spec, idx, radius=10):
    left = max(0, idx - radius)
    right = min(len(spec), idx + radius + 1)
    window = np.concatenate([spec[left:max(left, idx - 1)], spec[min(right, idx + 2):right]])
    if window.size == 0:
        return float(np.median(spec))
    return float(np.median(window))


def save_fault_peak_preservation(
    clean_spec,
    noisy_spec,
    denoised_spec,
    labels,
    path,
    metrics_csv=None,
    max_classes=9,
    top_k=5,
):
    try:
        import csv
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip fault peak preservation plot: {exc}")
        return []
    clean_spec = np.asarray(clean_spec).squeeze(1)
    noisy_spec = np.asarray(noisy_spec).squeeze(1)
    denoised_spec = np.asarray(denoised_spec).squeeze(1)
    labels = np.asarray(labels) if labels is not None else np.zeros(len(clean_spec), dtype=int)
    classes = np.unique(labels)[:max_classes]
    rows = []
    path = Path(path)
    ensure_dir(path.parent)
    fig, axes = plt.subplots(len(classes), 1, figsize=(11, max(2.2, 2.0 * len(classes))), squeeze=False)
    for ax, c in zip(axes[:, 0], classes):
        idx = np.where(labels == c)[0]
        c_clean = clean_spec[idx].mean(axis=0)
        c_noisy = noisy_spec[idx].mean(axis=0)
        c_den = denoised_spec[idx].mean(axis=0)
        peaks = _top_clean_peaks(c_clean, top_k=top_k)
        ax.plot(c_clean, label="clean", linewidth=1.1)
        ax.plot(c_noisy, label="noisy", linewidth=0.8, alpha=0.65)
        ax.plot(c_den, label="denoised", linewidth=1.0)
        for p in peaks:
            ax.axvline(p, color="black", linestyle="--", linewidth=0.6, alpha=0.35)
            ax.scatter([p], [c_clean[p]], s=16, color="black", zorder=4)
            clean_amp = float(c_clean[p])
            noisy_amp = float(c_noisy[p])
            den_amp = float(c_den[p])
            denom = max(abs(clean_amp), 1e-8)
            noisy_floor = _local_noise_floor(c_noisy, p)
            den_floor = _local_noise_floor(c_den, p)
            rows.append(
                {
                    "class": int(c) if np.asarray(c).dtype.kind in {"i", "u"} else str(c),
                    "peak_bin": int(p),
                    "clean_amp": clean_amp,
                    "noisy_amp": noisy_amp,
                    "denoised_amp": den_amp,
                    "noisy_relative_abs_error": abs(noisy_amp - clean_amp) / denom,
                    "denoised_relative_abs_error": abs(den_amp - clean_amp) / denom,
                    "noisy_peak_to_floor": noisy_amp / max(abs(noisy_floor), 1e-8),
                    "denoised_peak_to_floor": den_amp / max(abs(den_floor), 1e-8),
                }
            )
        ax.set_title(f"class {c}: clean-spectrum peak preservation")
        ax.set_xlabel("envelope-spectrum bin")
        ax.set_ylabel("normalized amplitude")
    axes[0, 0].legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved: {path.resolve()}")
    if metrics_csv is not None and rows:
        metrics_csv = Path(metrics_csv)
        ensure_dir(metrics_csv.parent)
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {metrics_csv.resolve()}")
    return rows


def save_comparison_bar(summary_csv, path):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"Skip comparison bar plot: {exc}")
        return
    df = pd.read_csv(summary_csv)
    metrics = [m for m in ["snr_improvement", "rmse", "envelope_corr", "latent_mse_to_clean", "inter_intra_ratio", "logistic_accuracy"] if m in df]
    if not metrics:
        return
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 2.5 * len(metrics)), squeeze=False)
    for ax, m in zip(axes[:, 0], metrics):
        ax.bar(df["version"], df[m])
        ax.set_title(m)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved: {path.resolve()}")
