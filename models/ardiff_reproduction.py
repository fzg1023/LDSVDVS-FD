"""Independent ARDiff paper implementation, with explicitly recorded assumptions.

Implements Figs. 3-5 and Eqs. 2, 4, 8-15 of DOI 10.1016/j.ymssp.2025.113726.
Unreported widths, band boundaries and normalization axes are not author code.
The previous ardiff_inspired modules and checkpoints are intentionally untouched.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.ardiff_equations import vsm_scale
from models.ardiff_inspired import TimeEmbedding, TimeResidual


class ChannelLayerNorm(nn.Module):
    """LayerNorm over feature channels at each time/frequency position."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class LinearComplexAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        if channels % heads:
            raise ValueError('channels must be divisible by heads')
        self.heads, self.head_dim = heads, channels // heads
        self.qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.project = nn.Conv1d(channels, channels, 1)
        self.output_norm = ChannelLayerNorm(channels)

    def forward(self, real, imag):
        b, c, n = real.shape
        qr, kr, vr = self.qkv(real).reshape(b, 3, self.heads, self.head_dim, n).unbind(1)
        qi, ki, vi = self.qkv(imag).reshape(b, 3, self.heads, self.head_dim, n).unbind(1)
        kr, ki = kr.softmax(-2), ki.softmax(-2)
        qr, qi = qr.softmax(-1), qi.softmax(-1)
        cr = kr @ vr.transpose(-1, -2) - ki @ vi.transpose(-1, -2)
        ci = kr @ vi.transpose(-1, -2) + ki @ vr.transpose(-1, -2)
        real_out = (cr @ qr - ci @ qi) / math.sqrt(self.head_dim)
        imag_out = (cr @ qi + ci @ qr) / math.sqrt(self.head_dim)
        return (self.output_norm(self.project(real_out.reshape(b, c, n))),
                self.output_norm(self.project(imag_out.reshape(b, c, n))))


class FrequencyAttentionModule(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.input_norm = ChannelLayerNorm(channels)
        self.bands = nn.ModuleList([LinearComplexAttention(channels, heads) for _ in range(3)])

    def forward(self, x):
        # Float32 FFT also supports non-power-of-two intermediate resolutions under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.rfft(self.input_norm(x.float()), norm='ortho')
        parts = torch.tensor_split(spectrum, 3, dim=-1)
        if any(part.shape[-1] == 0 for part in parts):
            raise ValueError('Signal is too short to form three nonempty frequency bands')
        filtered = []
        for module, part in zip(self.bands, parts):
            real, imag = module(part.real, part.imag)
            filtered.append(torch.complex(real.float(), imag.float()))
        with torch.autocast(device_type=x.device.type, enabled=False):
            restored = torch.fft.irfft(torch.cat(filtered, dim=-1), n=x.shape[-1], norm='ortho')
        # One outer residual follows the text; nested identity arrows in Fig. 4 are ambiguous.
        return x + restored.to(x.dtype)


class FullTimeAttention(nn.Module):
    """Exact full-resolution self-attention; no pooling or token reduction."""

    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads, self.head_dim = heads, channels // heads
        self.norm = ChannelLayerNorm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.project = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b, c, n = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, self.head_dim, n).unbind(1)
        attended = F.scaled_dot_product_attention(
            q.transpose(-1, -2).contiguous(), k.transpose(-1, -2).contiguous(),
            v.transpose(-1, -2).contiguous(), dropout_p=0.0)
        return x + self.project(attended.transpose(-1, -2).reshape(b, c, n))


class ARDiffStage(nn.Module):
    """Fig. 3: ResBlock -> FAM -> ResBlock -> Time Attention."""

    def __init__(self, in_channels, channels, time_width, heads):
        super().__init__()
        self.res_before = TimeResidual(in_channels, channels, time_width)
        self.fam = FrequencyAttentionModule(channels, heads)
        self.res_after = TimeResidual(channels, channels, time_width)
        self.time_attention = FullTimeAttention(channels, heads)

    def forward(self, x, embedding):
        x = self.res_before(x, embedding)
        x = self.fam(x)
        return self.time_attention(self.res_after(x, embedding))


class ARDiffPaperUNet(nn.Module):
    def __init__(self, width=32, levels=3, heads=4):
        super().__init__()
        if levels < 2 or width % 8 or width % heads:
            raise ValueError('Require levels>=2 and width divisible by 8 and heads')
        self.levels = levels
        channels = [width * 2**i for i in range(levels)]
        self.time = TimeEmbedding(width)
        self.stem = nn.Conv1d(1, width, 7, padding=3)
        self.encoder = nn.ModuleList([ARDiffStage(c, c, width * 4, heads) for c in channels])
        self.down = nn.ModuleList([nn.Conv1d(a, b, 4, stride=2, padding=1)
                                   for a, b in zip(channels[:-1], channels[1:])])
        self.up = nn.ModuleList([nn.ConvTranspose1d(channels[i], channels[i-1], 4, stride=2, padding=1)
                                 for i in range(levels-1, 0, -1)])
        self.decoder = nn.ModuleList([ARDiffStage(channels[i-1] * 2, channels[i-1], width * 4, heads)
                                      for i in range(levels-1, 0, -1)])
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv1d(width, 1, 3, padding=1))

    def forward(self, x, t):
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] % (2**(self.levels-1)):
            raise ValueError('Expected [B,1,L] with length divisible by all downsampling factors')
        embedding = self.time(t)
        x = self.stem(x)
        skips = []
        for i, stage in enumerate(self.encoder):
            x = stage(x, embedding)
            if i < len(self.down):
                skips.append(x)
                x = self.down[i](x)
        for up, stage, skip in zip(self.up, self.decoder, reversed(skips)):
            x = stage(torch.cat([up(x), skip], dim=1), embedding)
        return self.out(x)


class ARDiffStepPredictor(nn.Module):
    """Fig. 5 conv/linear blocks; the run plan defines scalar regression units."""

    def __init__(self):
        super().__init__()
        channels, kernels = [1, 64, 128, 256, 512, 1024], [7, 5, 5, 3, 3]
        blocks = []
        for ci, co, kernel in zip(channels[:-1], channels[1:], kernels):
            blocks.extend([nn.Conv1d(ci, co, kernel, padding=kernel//2), nn.BatchNorm1d(co),
                           nn.ReLU(), nn.MaxPool1d(2)])
        self.conv = nn.Sequential(*blocks)
        # Temporal reduction to the figure's 1024-dimensional FC input is unreported.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                nn.Linear(256, 1))

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).flatten(1)).squeeze(-1)


@dataclass(frozen=True)
class TrainMinMax:
    """One affine transform fitted on clean train only; no test clipping/refitting."""

    center: float
    half_range: float

    @classmethod
    def fit(cls, clean_train):
        if not torch.isfinite(clean_train).all():
            raise ValueError('Nonfinite clean training array')
        lo, hi = float(clean_train.min()), float(clean_train.max())
        if hi - lo < 1e-12:
            raise ValueError('Constant training data cannot define a Min-Max transform')
        return cls((lo + hi) / 2, (hi - lo) / 2)

    def encode(self, x):
        return (x - self.center) / self.half_range

    def decode(self, x):
        return x * self.half_range + self.center


def forward_sample(clean, alpha_bar, generator=None):
    t = torch.randint(1, len(alpha_bar) + 1, (len(clean),), device=clean.device, generator=generator)
    eps = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    a = alpha_bar[t-1].view(-1, 1, 1)
    return a.sqrt() * clean + (1-a).sqrt() * eps, t, eps


def rounded_steps(raw_prediction, total_steps):
    if not torch.isfinite(raw_prediction).all():
        raise ValueError('Nonfinite NSPM output')
    # Intermediate-step inference t<T; direct regression avoids the old affine offset.
    return raw_prediction.round().long().clamp(1, total_steps-1)


@torch.no_grad()
def reverse_denoise(model, predictor, noisy, alpha_bar, *, start_steps=None, amp=False,
                    step_prediction_scale=1.0):
    """Eq. 4 with sigma=0, VSM, and explicitly defined physical-amplitude restoration.

    `noisy` is in training affine units. No clean input or known SNR is needed.
    The final sqrt(alpha_start)/S gain undoes the signal scaling in Eq. 11.
    This final-gain convention is an implementation choice not specified in the PDF.
    """
    if not torch.isfinite(noisy).all():
        raise ValueError('Nonfinite noisy input')
    if not math.isfinite(step_prediction_scale) or step_prediction_scale <= 0:
        raise ValueError('step_prediction_scale must be finite and positive')
    autocast = lambda: torch.autocast('cuda', dtype=torch.float16) if amp and noisy.is_cuda else nullcontext()
    if start_steps is None:
        with autocast():
            prediction = predictor(noisy)
        starts = rounded_steps(prediction.float() * step_prediction_scale, len(alpha_bar))
    else:
        starts = torch.as_tensor(start_steps, device=noisy.device, dtype=torch.long).reshape(-1)
        if starts.numel() == 1:
            starts = starts.expand(len(noisy))
        if len(starts) != len(noisy) or torch.any((starts < 1) | (starts >= len(alpha_bar))):
            raise ValueError('start_steps must have batch length and satisfy 1<=t<T')
    a0 = alpha_bar[starts-1]
    x, scale, _ = vsm_scale(noisy.float(), a0)
    previous = torch.cat([alpha_bar.new_ones(1), alpha_bar[:-1]])
    for i in range(int(starts.max()), 0, -1):
        t = torch.full((len(x),), i, device=x.device, dtype=torch.long)
        with autocast():
            predicted_noise = model(x, t).float()
        at = alpha_bar[i-1] / previous[i-1]
        update = (x - (1-at) / (1-alpha_bar[i-1]).sqrt() * predicted_noise) / at.sqrt()
        x = torch.where((starts >= i).view(-1, 1, 1), update, x)
    restored = x * a0.sqrt().view(-1, 1, 1) / scale
    if not torch.isfinite(restored).all():
        raise ValueError('Nonfinite reverse diffusion output')
    return restored, starts
