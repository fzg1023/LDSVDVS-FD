import torch
from torch import nn
import torch.nn.functional as F
import math


class MultiScaleWaveletLikeBlock(nn.Module):
    def __init__(self, width, levels=3):
        super().__init__()
        self.levels = int(levels)
        self.low_convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1) for _ in range(self.levels)])
        self.high_convs = nn.ModuleList([nn.Conv1d(width, width, 3, padding=1) for _ in range(self.levels)])
        self.local = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.merge = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(8 if width % 8 == 0 else 1, width)

    def forward(self, x):
        length = x.shape[-1]
        acc = self.local(x)
        for level in range(self.levels):
            factor = 2 ** (level + 1)
            pooled = F.avg_pool1d(x, kernel_size=factor, stride=factor, ceil_mode=True)
            low = F.interpolate(self.low_convs[level](pooled), size=length, mode="linear", align_corners=False)
            coarse = F.interpolate(pooled, size=length, mode="linear", align_corners=False)
            high = self.high_convs[level](x - coarse)
            acc = acc + low + high
        return x + F.gelu(self.norm(self.merge(acc)))


class WNO1DDenoiser(nn.Module):
    def __init__(self, width=64, depth=4, levels=3, residual=True, residual_scale=1.0):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.stem = nn.Conv1d(1, width, 1)
        self.blocks = nn.Sequential(*[MultiScaleWaveletLikeBlock(width, levels) for _ in range(depth)])
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_residual=False):
        pred = self.head(self.blocks(self.stem(y)))
        if self.residual:
            x_hat = y - self.residual_scale * pred
            return (x_hat, pred) if return_residual else x_hat
        return (pred, y - pred) if return_residual else pred


def haar_dwt1d(x):
    if x.shape[-1] % 2:
        x = F.pad(x, (0, 1), mode="reflect")
    even = x[..., 0::2]
    odd = x[..., 1::2]
    scale = 1.0 / math.sqrt(2.0)
    low = (even + odd) * scale
    high = (even - odd) * scale
    return low, high


def haar_idwt1d(low, high, out_len=None):
    scale = 1.0 / math.sqrt(2.0)
    even = (low + high) * scale
    odd = (low - high) * scale
    x = torch.empty(*low.shape[:-1], low.shape[-1] * 2, device=low.device, dtype=low.dtype)
    x[..., 0::2] = even
    x[..., 1::2] = odd
    if out_len is not None:
        x = x[..., :out_len]
    return x


class HaarWaveletOperatorBlock(nn.Module):
    """A differentiable 1D WNO layer using explicit Haar DWT/IDWT coefficients."""

    def __init__(self, width, levels=3, kernel_size=1):
        super().__init__()
        self.levels = int(levels)
        self.low_kernel = nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2)
        self.high_kernels = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2) for _ in range(self.levels)]
        )
        self.local = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(8 if width % 8 == 0 else 1, width)

    def forward(self, x):
        original_lengths = []
        highs = []
        low = x
        for _ in range(self.levels):
            original_lengths.append(low.shape[-1])
            low, high = haar_dwt1d(low)
            highs.append(high)

        low = self.low_kernel(low)
        for level in reversed(range(self.levels)):
            high = self.high_kernels[level](highs[level])
            low = haar_idwt1d(low, high, out_len=original_lengths[level])

        wavelet_update = low
        local_update = self.local(x)
        update = self.pointwise(wavelet_update + local_update)
        return x + F.gelu(self.norm(update))


class TrueWNO1DDenoiser(nn.Module):
    def __init__(self, width=64, depth=4, levels=3, residual=True, residual_scale=1.0):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.stem = nn.Conv1d(1, width, 1)
        self.blocks = nn.Sequential(*[HaarWaveletOperatorBlock(width, levels) for _ in range(depth)])
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_residual=False):
        pred = self.head(self.blocks(self.stem(y)))
        if self.residual:
            x_hat = y - self.residual_scale * pred
            return (x_hat, pred) if return_residual else x_hat
        return (pred, y - pred) if return_residual else pred
