import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DenoiserUNet1D(nn.Module):
    def __init__(self, base_channels=32, mode="residual", residual_scale=1.0):
        super().__init__()
        if mode not in {"residual", "direct"}:
            raise ValueError("mode must be residual or direct.")
        self.mode = mode
        self.residual_scale = float(residual_scale)
        self.enc1 = ConvBlock(1, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)
        self.up2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, y, return_residual=False):
        original_len = y.shape[-1]
        pad = (4 - original_len % 4) % 4
        if pad:
            y_in = F.pad(y, (0, pad), mode="reflect")
        else:
            y_in = y
        e1 = self.enc1(y_in)
        e2 = self.enc2(F.avg_pool1d(e1, 2))
        e3 = self.enc3(F.avg_pool1d(e2, 2))
        b = self.bottleneck(e3)
        d2 = self.up2(b)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = F.interpolate(d2, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = F.interpolate(d1, size=e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        pred = self.out(d1)[..., :original_len]
        if self.mode == "residual":
            x_hat = y - self.residual_scale * pred
            return (x_hat, pred) if return_residual else x_hat
        return (pred, y - pred) if return_residual else pred
