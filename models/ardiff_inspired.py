"""Explicit, non-official 1-D implementation inspired by ARDiff.

Band boundaries, channel sizes and several training details are assumptions,
not claimed to reproduce the authors' unavailable source code.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width * 4))

    def forward(self, t):
        half = self.width // 2
        frequencies = torch.exp(torch.arange(half, device=t.device).float() * (-math.log(10000) / max(half - 1, 1)))
        angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return self.net(torch.cat((angles.sin(), angles.cos()), dim=1))


class TimeResidual(nn.Module):
    def __init__(self, ci, co, time_width):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ci)
        self.conv1 = nn.Conv1d(ci, co, 3, padding=1)
        self.time = nn.Linear(time_width, co)
        self.norm2 = nn.GroupNorm(8, co)
        self.conv2 = nn.Conv1d(co, co, 3, padding=1)
        self.skip = nn.Identity() if ci == co else nn.Conv1d(ci, co, 1)

    def forward(self, x, embedding):
        h = self.conv1(F.silu(self.norm1(x))) + self.time(F.silu(embedding)).unsqueeze(-1)
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class ComplexBandAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        if channels % heads:
            raise ValueError('channels must divide heads')
        self.heads, self.dim = heads, channels // heads
        self.qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.project = nn.Conv1d(channels, channels, 1)

    def forward(self, real, imag):
        batch, channels, n = real.shape
        qre, kre, vre = self.qkv(real).reshape(batch, 3, self.heads, self.dim, n).unbind(1)
        qim, kim, vim = self.qkv(imag).reshape(batch, 3, self.heads, self.dim, n).unbind(1)
        # Paper states key/channel and query/frequency softmax; contraction is an implementation choice.
        kre, kim = kre.softmax(dim=-2), kim.softmax(dim=-2)
        qre, qim = qre.softmax(dim=-1), qim.softmax(dim=-1)
        cr = torch.einsum('bhdn,bhen->bhde', kre, vre) - torch.einsum('bhdn,bhen->bhde', kim, vim)
        ci = torch.einsum('bhdn,bhen->bhde', kre, vim) + torch.einsum('bhdn,bhen->bhde', kim, vre)
        out_r = (torch.einsum('bhde,bhen->bhdn', cr, qre) - torch.einsum('bhde,bhen->bhdn', ci, qim)) / math.sqrt(self.dim)
        out_i = (torch.einsum('bhde,bhen->bhdn', cr, qim) + torch.einsum('bhde,bhen->bhdn', ci, qre)) / math.sqrt(self.dim)
        return self.project(out_r.reshape(batch, channels, n)), self.project(out_i.reshape(batch, channels, n))


class FrequencyAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.bands = nn.ModuleList([ComplexBandAttention(channels) for _ in range(3)])

    def forward(self, x):
        z = torch.fft.rfft(self.norm(x), dim=-1, norm='ortho')
        # Equal-width bands are an assumption, not established by the paper.
        pieces = torch.tensor_split(z, 3, dim=-1)
        result = []
        for module, part in zip(self.bands, pieces):
            re, im = module(part.real, part.imag)
            result.append(torch.complex(re, im))
        filtered = torch.cat(result, dim=-1)
        return x + torch.fft.irfft(filtered, n=x.shape[-1], dim=-1, norm='ortho')


class TimeAttention(nn.Module):
    def __init__(self, channels, heads=4, max_tokens=256):
        super().__init__()
        self.heads, self.dim, self.max_tokens = heads, channels // heads, max_tokens
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        batch, channels, length = x.shape
        tokens = min(length, self.max_tokens)
        h = F.adaptive_avg_pool1d(self.norm(x), tokens)
        q, k, v = self.qkv(h).reshape(batch, 3, self.heads, self.dim, tokens).unbind(1)
        h = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        h = self.proj(h.transpose(-1, -2).reshape(batch, channels, tokens))
        if tokens != length:
            h = F.interpolate(h, size=length, mode='linear', align_corners=False)
        return x + h


class ARDiffInspiredUNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.time = TimeEmbedding(width)
        tw = width * 4
        self.stem = nn.Conv1d(1, width, 7, padding=3)
        self.enc1, self.enc2, self.mid = TimeResidual(width, width, tw), TimeResidual(width*2, width*2, tw), TimeResidual(width*4, width*4, tw)
        self.fam1, self.fam2, self.fam_mid = FrequencyAttention(width), FrequencyAttention(width*2), FrequencyAttention(width*4)
        self.temporal1, self.temporal2, self.temporal_mid = TimeAttention(width), TimeAttention(width*2), TimeAttention(width*4)
        self.down1 = nn.Conv1d(width, width*2, 4, stride=2, padding=1)
        self.down2 = nn.Conv1d(width*2, width*4, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(width*4, width*2, 4, stride=2, padding=1)
        self.dec2 = TimeResidual(width*4, width*2, tw)
        self.up1 = nn.ConvTranspose1d(width*2, width, 4, stride=2, padding=1)
        self.dec1 = TimeResidual(width*2, width, tw)
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv1d(width, 1, 3, padding=1))

    def forward(self, x, t):
        if x.shape[-1] % 4:
            raise ValueError('Length must be divisible by 4')
        e = self.time(t)
        h1 = self.temporal1(self.fam1(self.enc1(self.stem(x), e)))
        h2 = self.temporal2(self.fam2(self.enc2(self.down1(h1), e)))
        h = self.temporal_mid(self.fam_mid(self.mid(self.down2(h2), e)))
        h = self.dec2(torch.cat((self.up2(h), h2), dim=1), e)
        h = self.dec1(torch.cat((self.up1(h), h1), dim=1), e)
        return self.out(h)


class NoiseStepPredictor(nn.Module):
    """Adapted from Fig. 5; global averaging is an unstated architecture choice."""
    def __init__(self):
        super().__init__()
        channels = [1, 64, 128, 256, 512, 1024]
        kernels = [7, 5, 5, 3, 3]
        blocks = []
        for ci, co, k in zip(channels[:-1], channels[1:], kernels):
            blocks += [nn.Conv1d(ci, co, k, padding=k//2), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(2)]
        self.conv = nn.Sequential(*blocks)
        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.fc(self.conv(x).mean(dim=-1)).squeeze(-1)
