import torch
from torch import nn
import torch.nn.functional as F


class ViT1DDenoiser(nn.Module):
    def __init__(
        self,
        width=128,
        depth=4,
        heads=4,
        patch_size=16,
        mlp_ratio=4,
        dropout=0.05,
        max_patches=512,
        residual=False,
        residual_scale=1.0,
    ):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.patch_size = int(patch_size)
        self.patch_embed = nn.Conv1d(1, width, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, width))
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
        self.patch_decode = nn.Linear(width, self.patch_size)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pos(self, n_tokens, device, dtype):
        if n_tokens <= self.pos_embed.shape[1]:
            return self.pos_embed[:, :n_tokens].to(device=device, dtype=dtype)
        pos = F.interpolate(self.pos_embed.transpose(1, 2), size=n_tokens, mode="linear", align_corners=False)
        return pos.transpose(1, 2).to(device=device, dtype=dtype)

    def _decode_tokens(self, tokens, out_len):
        patches = self.patch_decode(tokens)
        x = patches.reshape(patches.shape[0], -1).unsqueeze(1)
        return x[..., :out_len]

    def forward(self, y, return_aux=False, return_residual=False):
        original_len = y.shape[-1]
        pad = (self.patch_size - original_len % self.patch_size) % self.patch_size
        y_in = F.pad(y, (0, pad), mode="reflect") if pad else y
        tokens = self.patch_embed(y_in).transpose(1, 2)
        tokens = tokens + self._pos(tokens.shape[1], tokens.device, tokens.dtype)
        tokens = self.norm(self.encoder(tokens))
        pred = self._decode_tokens(tokens, original_len)
        if self.residual:
            x_hat = y - self.residual_scale * pred
            if return_aux:
                return x_hat, {}
            return (x_hat, pred) if return_residual else x_hat
        if return_aux:
            return pred, {}
        return (pred, y - pred) if return_residual else pred


class ViTPODBranchNet(nn.Module):
    def __init__(self, rank, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, width, 7, padding=3, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv1d(width, width * 2, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, width * 2),
            nn.GELU(),
            nn.Conv1d(width * 2, width * 2, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, width * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(width * 2, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, rank),
        )

    def forward(self, x):
        return self.net(x)


class PODViT1DDenoiser(nn.Module):
    def __init__(
        self,
        rank,
        width=128,
        depth=4,
        heads=4,
        patch_size=16,
        mlp_ratio=4,
        dropout=0.05,
        max_patches=512,
        residual=False,
        residual_scale=1.0,
    ):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.patch_size = int(patch_size)
        self.branch = ViTPODBranchNet(rank, width=64)
        self.cond_token = nn.Sequential(nn.Linear(rank, width), nn.GELU(), nn.Linear(width, width))
        self.patch_embed = nn.Conv1d(1, width, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches + 1, width))
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
        self.patch_decode = nn.Linear(width, self.patch_size)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _pos(self, n_tokens, device, dtype):
        if n_tokens <= self.pos_embed.shape[1]:
            return self.pos_embed[:, :n_tokens].to(device=device, dtype=dtype)
        pos = F.interpolate(self.pos_embed.transpose(1, 2), size=n_tokens, mode="linear", align_corners=False)
        return pos.transpose(1, 2).to(device=device, dtype=dtype)

    def _decode_tokens(self, tokens, out_len):
        patches = self.patch_decode(tokens)
        x = patches.reshape(patches.shape[0], -1).unsqueeze(1)
        return x[..., :out_len]

    def forward(self, y, return_aux=True):
        original_len = y.shape[-1]
        pad = (self.patch_size - original_len % self.patch_size) % self.patch_size
        y_in = F.pad(y, (0, pad), mode="reflect") if pad else y
        a_hat = self.branch(y)
        cond = self.cond_token(a_hat).unsqueeze(1)
        tokens = self.patch_embed(y_in).transpose(1, 2)
        tokens = torch.cat([cond, tokens], dim=1)
        tokens = tokens + self._pos(tokens.shape[1], tokens.device, tokens.dtype)
        tokens = self.norm(self.encoder(tokens))
        pred = self._decode_tokens(tokens[:, 1:], original_len)
        x_hat = y - self.residual_scale * pred if self.residual else pred
        aux = {"a_hat": a_hat}
        return (x_hat, aux) if return_aux else x_hat
