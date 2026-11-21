import numpy as np
import torch
from run_pod_denoising_experiments import normalize_batch

def add_noise(clean, snrs, offset=0, base=190000):
    a = clean.detach().cpu().numpy().astype(np.float64)
    envelope = np.linspace(0.2, 1.8, a.shape[-1])
    out = []
    for i, x in enumerate(a):
        snr = snrs[(i + offset) % len(snrs)] if isinstance(snrs, list) else snrs
        rng = np.random.default_rng(base + offset + i)
        n = rng.standard_normal(x.shape) * envelope
        n *= np.sqrt(np.mean(x*x) / (10**(float(snr)/10) * np.mean(n*n)))
        out.append(x+n)
    return torch.from_numpy(np.stack(out)).to(clean.device).float()


def noise_for_model(clean, snrs, config, seed_offset=0):
    noisy = add_noise(clean, snrs, seed_offset, base=190000 if isinstance(snrs, list) else 290000)
    norm, meta = normalize_batch(noisy, 'rms')
    return clean / meta['scale'], norm, noisy, meta

