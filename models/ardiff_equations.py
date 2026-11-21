"""Verifiable ARDiff equations; not a complete ARDiff network reproduction.

The standard cosine offset and beta clipping below are explicit implementation
choices; the paper identifies a cosine scheduler but does not specify both here.
See Kim et al., MSSP 244 (2026), 113726, Eqs. 2 and 11-15.
"""
import math
import torch


def cosine_alpha_bar(steps=1000, offset=0.008, max_beta=0.999, *, device=None, dtype=torch.float64):
    if steps < 2 or offset < 0 or not 0 < max_beta < 1:
        raise ValueError('Invalid schedule parameters')
    t = torch.linspace(0, steps, steps + 1, device=device, dtype=dtype) / steps
    f = torch.cos((t + offset) / (1 + offset) * math.pi / 2).square()
    f = f / f[0]
    betas = (1 - f[1:] / f[:-1]).clamp(0, max_beta)
    return torch.cumprod(1 - betas, dim=0)


def nearest_step_for_snr(snr_db, alpha_bar):
    snr = torch.as_tensor(snr_db, dtype=alpha_bar.dtype, device=alpha_bar.device)
    target = torch.sigmoid(snr * (math.log(10.0) / 10.0))
    indices = (alpha_bar.reshape(1, -1) - target.reshape(-1, 1)).abs().argmin(dim=1)
    return (indices + 1).reshape(snr.shape)


def vsm_scale(noisy, alpha_bar, eps=1e-8):
    """Paper Eqs. 12-15; scale uses noisy input and predicted cumulative alpha.

    Assumes additive uncorrelated signal and noise. Returns the scaled input,
    scale and estimated clean variance; no clean target is used at inference.
    """
    if not torch.isfinite(noisy).all():
        raise ValueError('Noisy signal contains NaN/Inf')
    alpha_bar = torch.as_tensor(alpha_bar, dtype=noisy.dtype, device=noisy.device)
    if torch.any((alpha_bar <= 0) | (alpha_bar >= 1)):
        raise ValueError('alpha_bar must be strictly between zero and one')
    while alpha_bar.ndim < noisy.ndim:
        alpha_bar = alpha_bar.unsqueeze(-1)
    var_y = noisy.var(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    # Eq. 13 simplifies to alpha_bar * var_y using Eq. 12.
    var_clean_est = alpha_bar * var_y
    var_diffused = alpha_bar * var_clean_est + (1 - alpha_bar)
    scale = (var_diffused / var_y).sqrt()
    return noisy * scale, scale, var_clean_est
