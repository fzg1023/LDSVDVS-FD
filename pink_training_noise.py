import numpy as np
import torch
from run_pod_denoising_experiments import normalize_batch

def pink_batch(clean, snrs, seed_offset=0, base_seed=190000):
    a = clean.detach().cpu().numpy().astype(np.float64)
    out = []
    f = np.fft.rfftfreq(a.shape[-1], 1 / 25600)
    weight = np.zeros_like(f)
    weight[1:] = 1 / np.sqrt(f[1:])
    for i, x in enumerate(a):
        snr = snrs[(i + seed_offset) % len(snrs)] if isinstance(snrs, list) else snrs
        rng = np.random.default_rng(base_seed + seed_offset + i)
        n = np.fft.irfft(np.fft.rfft(rng.standard_normal(x.shape), axis=-1) * weight, n=x.shape[-1], axis=-1)
        n *= np.sqrt(np.mean(x*x) / (10**(float(snr)/10) * np.mean(n*n)))
        out.append(x + n)
    return torch.from_numpy(np.stack(out)).to(clean.device).float()


def make_pink_for_model(clean, snrs, config, seed_offset=0):
    noisy = pink_batch(clean, snrs, seed_offset, base_seed=190000 if isinstance(snrs, list) else 290000)
    noisy_model, meta = normalize_batch(noisy, 'rms')
    return clean / meta['scale'], noisy_model, noisy, meta

