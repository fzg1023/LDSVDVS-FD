import torch
from torch import nn
import torch.nn.functional as F

from transforms.frequency import batch_frequency_transform


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        groups = min(8, out_ch)
        while out_ch % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class FrequencyManifoldEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, spectrum):
        return self.net(spectrum)


class FilmLayer(nn.Module):
    def __init__(self, latent_dim, channels):
        super().__init__()
        self.to_gamma_beta = nn.Linear(latent_dim, channels * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, feat, z):
        gb = self.to_gamma_beta(z).unsqueeze(-1)
        gamma, beta = gb.chunk(2, dim=1)
        return feat * (1.0 + gamma) + beta, gamma.detach(), beta.detach()


class GateLayer(nn.Module):
    def __init__(self, latent_dim, channels):
        super().__init__()
        self.to_gate = nn.Linear(latent_dim, channels)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.zeros_(self.to_gate.bias)

    def forward(self, feat, z):
        gate = torch.sigmoid(self.to_gate(z)).unsqueeze(-1)
        return feat * gate, gate.detach()


class FMEDNet(nn.Module):
    def __init__(
        self,
        base_channels=32,
        depth=4,
        latent_dim=64,
        kernel_size=3,
        fusion_mode="film",
        freq_mode="envelope",
        freq_normalize=True,
        residual_denoising=True,
        residual_scale=1.0,
        concat_fusion=False,
    ):
        super().__init__()
        if depth != 4:
            raise ValueError("FMEDNet currently implements depth=4 to match the requested U-Net layout.")
        if fusion_mode not in {"bottleneck", "film", "gate"}:
            raise ValueError("fusion_mode must be bottleneck, film, or gate.")
        self.fusion_mode = fusion_mode
        self.freq_mode = freq_mode
        self.freq_normalize = bool(freq_normalize)
        self.residual_denoising = bool(residual_denoising)
        self.residual_scale = float(residual_scale)
        self.concat_fusion = bool(concat_fusion)

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = ConvBlock1D(1, c1, kernel_size)
        self.enc2 = ConvBlock1D(c1, c2, kernel_size)
        self.enc3 = ConvBlock1D(c2, c3, kernel_size)
        self.enc4 = ConvBlock1D(c3, c4, kernel_size)
        self.e_m = FrequencyManifoldEncoder(latent_dim)

        self.bottleneck_proj = nn.Sequential(nn.Linear(latent_dim, c4), nn.GELU(), nn.Linear(c4, c4))
        self.bottleneck_reduce = nn.Conv1d(c4 * 2, c4, 1) if concat_fusion else None
        self.up3 = nn.ConvTranspose1d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock1D(c3 + c3, c3, kernel_size)
        self.up2 = nn.ConvTranspose1d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock1D(c2 + c2, c2, kernel_size)
        self.up1 = nn.ConvTranspose1d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock1D(c1 + c1, c1, kernel_size)
        self.out = nn.Conv1d(c1, 1, 1)

        self.film3 = FilmLayer(latent_dim, c3)
        self.film2 = FilmLayer(latent_dim, c2)
        self.film1 = FilmLayer(latent_dim, c1)
        self.gate3 = GateLayer(latent_dim, c3)
        self.gate2 = GateLayer(latent_dim, c2)
        self.gate1 = GateLayer(latent_dim, c1)

    def _align(self, x, ref):
        if x.shape[-1] == ref.shape[-1]:
            return x
        return F.interpolate(x, size=ref.shape[-1], mode="linear", align_corners=False)

    def _modulate(self, feat, z, film, gate, aux, name):
        if self.fusion_mode == "film":
            feat, gamma, beta = film(feat, z)
            aux.setdefault("gamma_beta", {})[name] = {"gamma": gamma, "beta": beta}
        elif self.fusion_mode == "gate":
            feat, g = gate(feat, z)
            aux.setdefault("gate_values", {})[name] = g
        return feat

    def forward(self, y, return_aux=True):
        original_len = y.shape[-1]
        pad = (8 - original_len % 8) % 8
        y_in = F.pad(y, (0, pad), mode="reflect") if pad else y
        s_y = batch_frequency_transform(y, mode=self.freq_mode, normalize=self.freq_normalize)
        z_m = self.e_m(s_y)

        e1 = self.enc1(y_in)
        e2 = self.enc2(F.avg_pool1d(e1, 2))
        e3 = self.enc3(F.avg_pool1d(e2, 2))
        h = self.enc4(F.avg_pool1d(e3, 2))
        aux = {"z_m": z_m, "time_feat": h}

        g = self.bottleneck_proj(z_m).unsqueeze(-1).expand(-1, -1, h.shape[-1])
        if self.concat_fusion:
            h = self.bottleneck_reduce(torch.cat([h, g], dim=1))
        else:
            h = h + g

        d3 = self._align(self.up3(h), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d3 = self._modulate(d3, z_m, self.film3, self.gate3, aux, "dec3")
        d2 = self._align(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d2 = self._modulate(d2, z_m, self.film2, self.gate2, aux, "dec2")
        d1 = self._align(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = self._modulate(d1, z_m, self.film1, self.gate1, aux, "dec1")
        pred = self.out(d1)[..., :original_len]
        x_hat = y - self.residual_scale * pred if self.residual_denoising else pred
        return (x_hat, aux) if return_aux else x_hat
