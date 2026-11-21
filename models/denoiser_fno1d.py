import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        scale = 1.0 / max(1, in_channels * out_channels)
        weight = scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"SpectralConv1d expects [B,C,L], got {tuple(x.shape)}")
        b, _, length = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        available = x_ft.shape[-1]
        modes = min(self.modes, available)
        out_ft = torch.zeros(b, self.out_channels, available, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :modes], self.weight[:, :, :modes])
        return torch.fft.irfft(out_ft, n=length, dim=-1)


class FNOBlock1D(nn.Module):
    def __init__(self, width, modes, dropout=0.0, use_norm=True):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(8 if width % 8 == 0 else 1, width) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.spectral(x) + self.pointwise(x)
        y = self.dropout(F.gelu(self.norm(y)))
        return x + y


class FNO1DDenoiser(nn.Module):
    def __init__(self, width=64, modes=32, depth=4, dropout=0.0, residual=True, residual_scale=1.0):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.lift = nn.Conv1d(1, width, kernel_size=1)
        self.blocks = nn.Sequential(*[FNOBlock1D(width, modes, dropout=dropout) for _ in range(depth)])
        self.project = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_residual=False, return_aux=False):
        h = self.lift(y)
        h = self.blocks(h)
        pred = self.project(h)
        if self.residual:
            x_hat = y - self.residual_scale * pred
            if return_aux:
                return x_hat, {}
            return (x_hat, pred) if return_residual else x_hat
        if return_aux:
            return pred, {}
        return (pred, y - pred) if return_residual else pred
