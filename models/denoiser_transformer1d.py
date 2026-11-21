import math

import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding1D(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        if x.shape[1] <= self.pe.shape[1]:
            return x + self.pe[:, : x.shape[1]].to(dtype=x.dtype)
        pos = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, x.shape[2], 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / x.shape[2]))
        pe = torch.zeros(1, x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[: pe[:, :, 1::2].shape[-1]])
        return x + pe


class Transformer1DDenoiser(nn.Module):
    def __init__(
        self,
        width=128,
        depth=4,
        heads=4,
        patch_size=16,
        mlp_ratio=4,
        dropout=0.05,
        residual=False,
        residual_scale=1.0,
    ):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.patch_size = int(patch_size)
        self.patch = nn.Conv1d(1, width, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos = PositionalEncoding1D(width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Conv1d(width, width, 3, padding=1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_aux=False, return_residual=False):
        original_len = y.shape[-1]
        pad = (self.patch_size - original_len % self.patch_size) % self.patch_size
        y_in = F.pad(y, (0, pad), mode="reflect") if pad else y
        tokens = self.patch(y_in).transpose(1, 2)
        tokens = self.encoder(self.pos(tokens))
        feat = self.norm(tokens).transpose(1, 2)
        feat = F.interpolate(feat, size=y_in.shape[-1], mode="linear", align_corners=False)
        pred = self.head(feat)[..., :original_len]
        if self.residual:
            x_hat = y - self.residual_scale * pred
            if return_aux:
                return x_hat, {}
            return (x_hat, pred) if return_residual else x_hat
        if return_aux:
            return pred, {}
        return (pred, y - pred) if return_residual else pred
