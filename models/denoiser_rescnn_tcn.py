import torch
from torch import nn
import torch.nn.functional as F


class ResidualConvBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=1, dropout=0.0):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
        )

    def forward(self, x):
        return x + F.gelu(self.net(x))


class ResCNN1DDenoiser(nn.Module):
    def __init__(self, width=64, depth=8, kernel_size=7, residual=False, residual_scale=1.0):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.stem = nn.Sequential(nn.Conv1d(1, width, kernel_size, padding=kernel_size // 2), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualConvBlock1D(width, kernel_size=kernel_size) for _ in range(depth)])
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_aux=False, return_residual=False):
        pred = self.head(self.blocks(self.stem(y)))
        if self.residual:
            x_hat = y - self.residual_scale * pred
            if return_aux:
                return x_hat, {}
            return (x_hat, pred) if return_residual else x_hat
        if return_aux:
            return pred, {}
        return (pred, y - pred) if return_residual else pred


class TCNBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.05):
        super().__init__()
        padding = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _causal_trim(self, x, target_len):
        return x[..., :target_len]

    def forward(self, x):
        length = x.shape[-1]
        y = self._causal_trim(self.conv1(x), length)
        y = self.dropout(F.gelu(self.norm1(y)))
        y = self._causal_trim(self.conv2(y), length)
        y = self.dropout(F.gelu(self.norm2(y)))
        return x + y


class TCN1DDenoiser(nn.Module):
    def __init__(self, width=64, depth=8, kernel_size=5, residual=False, residual_scale=1.0, dropout=0.05):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.stem = nn.Conv1d(1, width, 1)
        self.blocks = nn.Sequential(
            *[TCNBlock1D(width, kernel_size=kernel_size, dilation=2 ** (i % 6), dropout=dropout) for i in range(depth)]
        )
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_aux=False, return_residual=False):
        pred = self.head(self.blocks(self.stem(y)))
        if self.residual:
            x_hat = y - self.residual_scale * pred
            if return_aux:
                return x_hat, {}
            return (x_hat, pred) if return_residual else x_hat
        if return_aux:
            return pred, {}
        return (pred, y - pred) if return_residual else pred
