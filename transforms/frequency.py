import warnings

import torch
import torch.nn.functional as F


def fft_magnitude(x):
    spectrum = torch.fft.rfft(x.squeeze(1), dim=-1)
    return torch.log1p(torch.abs(spectrum)).unsqueeze(1)


def _hilbert_envelope_torch(x):
    n = x.shape[-1]
    Xf = torch.fft.fft(x.squeeze(1), dim=-1)
    h = torch.zeros(n, device=x.device, dtype=x.dtype)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    analytic = torch.fft.ifft(Xf * h, dim=-1)
    return torch.abs(analytic).unsqueeze(1)


def envelope_spectrum(x):
    try:
        envelope = _hilbert_envelope_torch(x)
    except Exception as exc:
        warnings.warn(f"Hilbert envelope failed, falling back to FFT magnitude: {exc}")
        return fft_magnitude(x)
    spectrum = torch.fft.rfft(envelope.squeeze(1), dim=-1)
    return torch.log1p(torch.abs(spectrum)).unsqueeze(1)


def _linear_sample_last_dim(values, positions):
    max_pos = values.shape[-1] - 1
    positions = positions.clamp(0, max_pos)
    left = torch.floor(positions).long()
    right = (left + 1).clamp_max(max_pos)
    weight = (positions - left.to(positions.dtype)).view(1, 1, -1)
    left_values = values.gather(-1, left.view(1, 1, -1).expand(values.shape[0], values.shape[1], -1))
    right_values = values.gather(-1, right.view(1, 1, -1).expand(values.shape[0], values.shape[1], -1))
    return left_values * (1.0 - weight) + right_values * weight


def envelope_order_spectrum(x, fs=25600.0, shaft_speed_hz=20.0, max_order=80.0, order_bins=None):
    spectrum = envelope_spectrum(x)
    n = x.shape[-1]
    n_freq = spectrum.shape[-1]
    bins = int(order_bins or n_freq)
    order_axis = torch.linspace(0.0, float(max_order), bins, device=x.device, dtype=x.dtype)
    freq_axis = order_axis * float(shaft_speed_hz)
    df = float(fs) / float(n)
    positions = freq_axis / df
    return _linear_sample_last_dim(spectrum, positions)


def normalize_spectrum(spectrum, eps=1e-8):
    mean = spectrum.mean(dim=-1, keepdim=True)
    std = spectrum.std(dim=-1, keepdim=True).clamp_min(eps)
    return (spectrum - mean) / std


def batch_frequency_transform(
    x,
    mode="envelope",
    normalize=True,
    eps=1e-8,
    fs=25600.0,
    shaft_speed_hz=20.0,
    max_order=80.0,
    order_bins=None,
):
    if x.ndim != 3 or x.shape[1] != 1:
        raise ValueError(f"Expected input [B,1,L], got {tuple(x.shape)}.")
    if mode == "envelope":
        spectrum = envelope_spectrum(x)
    elif mode == "envelope_order":
        spectrum = envelope_order_spectrum(
            x,
            fs=fs,
            shaft_speed_hz=shaft_speed_hz,
            max_order=max_order,
            order_bins=order_bins,
        )
    elif mode == "fft":
        spectrum = fft_magnitude(x)
    else:
        raise ValueError(f"Unknown frequency mode: {mode}.")
    return normalize_spectrum(spectrum, eps) if normalize else spectrum


def resize_frequency_weight(weight, target_len):
    if weight.ndim == 1:
        weight = weight[None, None, :]
    elif weight.ndim == 2:
        weight = weight[:, None, :]
    return F.interpolate(weight, size=target_len, mode="linear", align_corners=False)
