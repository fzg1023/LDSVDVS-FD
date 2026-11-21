"""One-dimensional, reduced-width adaptation of the official Xformer.

Reference: gladzhang/Xformer, ecc1e910b923f35c8d46f5ffbc8ed2bc603f6567,
basicsr/models/archs/x_former_arch.py (ICLR 2024).
Retains two branches, four BCUs, shared temporal bottleneck, and residual output.
This is not the original image model or an exact paper-training reproduction.
"""

import torch
from torch import nn
from torch.nn import functional as F


class FeatureNorm(nn.Module):
    def __init__(self, channels, bias_free=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = None if bias_free else nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        y = x.transpose(1, 2)
        variance = y.var(-1, keepdim=True, unbiased=False)
        if self.bias is not None:
            y = y - y.mean(-1, keepdim=True)
        y = y * torch.rsqrt(variance + 1e-5) * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y.transpose(1, 2)


class ChannelAttention(nn.Module):
    def __init__(self, channels, heads, bias=False):
        super().__init__()
        if channels % heads:
            raise ValueError("Channels must be divisible by heads")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv1d(channels, channels * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv1d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=bias)
        self.project_out = nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, x):
        b, c, length = x.shape
        q, k, v = [y.reshape(b, self.heads, c // self.heads, length)
                   for y in self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attention = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(-1)
        return self.project_out((attention @ v).reshape(b, c, length))


class GatedFeedForward(nn.Module):
    def __init__(self, channels, expansion=2.66, bias=False):
        super().__init__()
        hidden = int(channels * expansion)
        self.project_in = nn.Conv1d(channels, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv1d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv1d(hidden, channels, 1, bias=bias)

    def forward(self, x):
        a, b = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(a) * b)


class ChannelBlock(nn.Module):
    def __init__(self, channels, heads, expansion=2.66):
        super().__init__()
        self.norm1, self.norm2 = FeatureNorm(channels), FeatureNorm(channels)
        self.attn = ChannelAttention(channels, heads)
        self.ffn = GatedFeedForward(channels, expansion)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class WindowAttention(nn.Module):
    def __init__(self, channels, heads, window):
        super().__init__()
        self.heads, self.window = heads, window
        self.scale = (channels // heads) ** -0.5
        self.qkv, self.proj = nn.Linear(channels, channels * 3), nn.Linear(channels, channels)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * window - 1, heads))
        positions = torch.arange(window)
        index = positions[:, None] - positions[None, :] + window - 1
        self.register_buffer("relative_position_index", index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        b, length, c = x.shape
        qkv = self.qkv(x).reshape(b, length, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(length, length, self.heads)
        scores = scores + bias.permute(2, 0, 1)[None]
        if mask is not None:
            windows = mask.shape[0]
            scores = (scores.reshape(b // windows, windows, self.heads, length, length)
                      + mask[None, :, None]).reshape(b, self.heads, length, length)
        return self.proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(b, length, c))


class DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = (keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device, dtype=x.dtype)).floor()
        return x * mask / keep


class TemporalBlock(nn.Module):
    def __init__(self, channels, heads, window=32, shift=0, expansion=2.66, drop_path=0.0):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.attn = WindowAttention(channels, heads, window)
        hidden = int(channels * expansion)
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels))
        self.drop = DropPath(drop_path)

    def shift_mask(self, length, device, dtype):
        region = torch.zeros(length, device=device, dtype=torch.long)
        region[-self.window:-self.shift] = 1
        region[-self.shift:] = 2
        region = region.reshape(-1, self.window)
        return (region[:, :, None] != region[:, None, :]).to(dtype) * -100.0

    def forward(self, x):
        b, c, length = x.shape
        if length % self.window:
            raise ValueError("Temporal length must be divisible by configured window")
        shortcut = x.transpose(1, 2)
        y = self.norm1(shortcut)
        shift = self.shift if length > self.window else 0
        if shift:
            y = torch.roll(y, -shift, dims=1)
        windows = y.reshape(-1, self.window, c)
        mask = self.shift_mask(length, y.device, y.dtype) if shift else None
        y = self.attn(windows, mask).reshape(b, length, c)
        if shift:
            y = torch.roll(y, shift, dims=1)
        y = shortcut + self.drop(y)
        return (y + self.drop(self.mlp(self.norm2(y)))).transpose(1, 2)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # A 1D factor-two unshuffle doubles channels; no 2D channel-halving step.
        self.conv = nn.Conv1d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        b, c, length = x.shape
        return x.reshape(b, c, length // 2, 2).permute(0, 1, 3, 2).reshape(b, c * 2, length // 2)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        b, c, length = x.shape
        return x.reshape(b, c // 2, 2, length).permute(0, 1, 3, 2).reshape(b, c // 2, length * 2)


class Xformer1D(nn.Module):
    def __init__(self, dim=24, channel_blocks=(1, 2, 2), temporal_blocks=(2, 2, 2, 2),
                 refinement_blocks=2, heads=(1, 2, 4, 8), window=32, expansion=2.66, drop_path=0.1):
        super().__init__()
        if len(channel_blocks) != 3 or len(temporal_blocks) != 4 or any(n < 1 for n in (*channel_blocks, *temporal_blocks)):
            raise ValueError("Expected three channel and four temporal stages")
        self.required_multiple = 8 * window
        widths = [dim * 2 ** i for i in range(4)]
        self.embed = nn.Conv1d(1, dim, 3, padding=1, bias=False)
        probabilities = torch.linspace(0, drop_path, sum(temporal_blocks)).tolist()

        def channel(stage, width=None):
            return nn.Sequential(*[ChannelBlock(width or widths[stage], heads[stage], expansion)
                                   for _ in range(channel_blocks[stage])])

        def temporal(stage):
            offset = sum(temporal_blocks[:stage])
            return nn.Sequential(*[TemporalBlock(widths[stage], heads[stage], window,
                0 if i % 2 == 0 else window // 2, expansion, probabilities[offset + i])
                for i in range(temporal_blocks[stage])])

        self.c_encoder = nn.ModuleList([channel(i) for i in range(3)])
        self.t_encoder = nn.ModuleList([temporal(i) for i in range(3)])
        self.c_down = nn.ModuleList([Downsample(widths[i]) for i in range(3)])
        self.t_down = nn.ModuleList([Downsample(widths[i]) for i in range(3)])
        # Official implementation applies the same spatial bottleneck to both branches.
        self.shared_latent = temporal(3)
        self.c_up = nn.ModuleList([Upsample(widths[i + 1]) for i in range(3)])
        self.t_up = nn.ModuleList([Upsample(widths[i + 1]) for i in range(3)])
        self.c_reduce = nn.ModuleList([nn.Conv1d(widths[i] * 2, widths[i], 1, bias=False) for i in range(3)])
        self.t_reduce = nn.ModuleList([nn.Conv1d(widths[i] * 2, widths[i], 1, bias=False) for i in range(3)])
        self.c_decoder = nn.ModuleList([channel(i) for i in range(3)])
        self.t_decoder = nn.ModuleList([temporal(i) for i in range(3)])
        bcu_widths = [widths[1], widths[2], widths[1], widths[0]]
        self.bcu_conv = nn.ModuleList([nn.Conv1d(c, c, 3, padding=1) for c in bcu_widths])
        self.bcu_depthwise = nn.ModuleList([nn.Conv1d(c, c, 3, padding=1, groups=c) for c in bcu_widths])
        self.refinement = nn.Sequential(*[ChannelBlock(dim * 2, heads[0], expansion) for _ in range(refinement_blocks)])
        self.output = nn.Conv1d(dim * 2, 1, 3, padding=1, bias=False)

    def fuse(self, channel, temporal, index):
        return channel + self.bcu_depthwise[index](temporal), temporal + self.bcu_conv[index](channel)

    def forward(self, x, return_aux=False):
        if x.ndim != 3 or x.shape[1] != 1 or x.shape[-1] % self.required_multiple:
            raise ValueError(f"Expected [B,1,L] with L divisible by {self.required_multiple}")
        c = t = self.embed(x)
        c_skips, t_skips = [], []
        for i in range(3):
            if i:
                c, t = self.fuse(c, t, i - 1)
            c, t = self.c_encoder[i](c), self.t_encoder[i](t)
            c_skips.append(c)
            t_skips.append(t)
            c, t = self.c_down[i](c), self.t_down[i](t)
        c, t = self.shared_latent(c), self.shared_latent(t)
        for i in [2, 1, 0]:
            c = self.c_reduce[i](torch.cat([self.c_up[i](c), c_skips[i]], dim=1))
            t = self.t_reduce[i](torch.cat([self.t_up[i](t), t_skips[i]], dim=1))
            if i < 2:
                c, t = self.fuse(c, t, 3 - i)
            c, t = self.c_decoder[i](c), self.t_decoder[i](t)
        result = x + self.output(self.refinement(torch.cat([c, t], dim=1)))
        return (result, {}) if return_aux else result
