import torch
from torch import nn
import torch.nn.functional as F

from models.fmed_net import ConvBlock1D, FilmLayer, GateLayer
from models.denoiser_fno1d import FNOBlock1D
from models.denoiser_rescnn_tcn import ResidualConvBlock1D, TCNBlock1D
from models.denoiser_transformer1d import PositionalEncoding1D
from models.denoiser_wno1d import HaarWaveletOperatorBlock, MultiScaleWaveletLikeBlock


class PODBranchNet(nn.Module):
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

    def forward(self, y):
        return self.net(y)


class ClassWisePODBranchNet(nn.Module):
    def __init__(self, num_classes, rank, width=64):
        super().__init__()
        self.num_classes = int(num_classes)
        self.rank = int(rank)
        self.features = nn.Sequential(
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
        )
        self.class_head = nn.Linear(width * 2, self.num_classes)
        self.coeff_head = nn.Linear(width * 2, self.num_classes * self.rank)

    def forward(self, y):
        feat = self.features(y)
        class_logits = self.class_head(feat)
        class_prob = torch.softmax(class_logits, dim=1)
        coeff_all = self.coeff_head(feat).view(y.shape[0], self.num_classes, self.rank)
        coeff_mix = torch.sum(class_prob.unsqueeze(-1) * coeff_all, dim=1)
        condition = torch.cat([class_prob, coeff_mix], dim=1)
        return class_logits, class_prob, coeff_all, coeff_mix, condition


class PODCoeffPredictor(nn.Module):
    def __init__(self, rank, width=64):
        super().__init__()
        self.branch = PODBranchNet(rank, width)

    def forward(self, y, return_aux=True):
        a_hat = self.branch(y)
        aux = {"a_hat": a_hat}
        return (None, aux) if return_aux else a_hat


class PODDeepONetTimeDenoiser(nn.Module):
    def __init__(self, mean, basis, width=64):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("basis", torch.as_tensor(basis, dtype=torch.float32))
        self.branch = PODBranchNet(self.basis.shape[0], width)

    def forward(self, y, return_aux=True):
        a_hat = self.branch(y)
        x_hat = self.mean + torch.matmul(a_hat, self.basis).unsqueeze(1)
        if x_hat.shape[-1] != y.shape[-1]:
            x_hat = F.interpolate(x_hat, size=y.shape[-1], mode="linear", align_corners=False)
        aux = {"a_hat": a_hat}
        return (x_hat, aux) if return_aux else x_hat


class PODFMEDNet(nn.Module):
    def __init__(
        self,
        rank,
        spectrum_len,
        base_channels=32,
        fusion_mode="film",
        residual_denoising=True,
        residual_scale=1.0,
        kernel_size=3,
    ):
        super().__init__()
        if fusion_mode not in {"bottleneck", "film", "gate"}:
            raise ValueError("fusion_mode must be bottleneck, film, or gate.")
        self.fusion_mode = fusion_mode
        self.residual_denoising = bool(residual_denoising)
        self.residual_scale = float(residual_scale)
        self.branch = PODBranchNet(rank, width=64)
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = ConvBlock1D(1, c1, kernel_size)
        self.enc2 = ConvBlock1D(c1, c2, kernel_size)
        self.enc3 = ConvBlock1D(c2, c3, kernel_size)
        self.enc4 = ConvBlock1D(c3, c4, kernel_size)
        self.bottleneck_proj = nn.Sequential(nn.Linear(rank, c4), nn.GELU(), nn.Linear(c4, c4))
        self.up3 = nn.ConvTranspose1d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock1D(c3 + c3, c3, kernel_size)
        self.up2 = nn.ConvTranspose1d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock1D(c2 + c2, c2, kernel_size)
        self.up1 = nn.ConvTranspose1d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock1D(c1 + c1, c1, kernel_size)
        self.out = nn.Conv1d(c1, 1, 1)
        self.film3, self.film2, self.film1 = FilmLayer(rank, c3), FilmLayer(rank, c2), FilmLayer(rank, c1)
        self.gate3, self.gate2, self.gate1 = GateLayer(rank, c3), GateLayer(rank, c2), GateLayer(rank, c1)

    def _align(self, x, ref):
        if x.shape[-1] == ref.shape[-1]:
            return x
        return F.interpolate(x, size=ref.shape[-1], mode="linear", align_corners=False)

    def _modulate(self, feat, a_hat, film, gate, aux, name):
        if self.fusion_mode == "film":
            feat, gamma, beta = film(feat, a_hat)
            aux.setdefault("gamma_beta", {})[name] = {"gamma": gamma, "beta": beta}
        elif self.fusion_mode == "gate":
            feat, g = gate(feat, a_hat)
            aux.setdefault("gate_values", {})[name] = g
        return feat

    def forward(self, y, return_aux=True):
        original_len = y.shape[-1]
        pad = (8 - original_len % 8) % 8
        y_in = F.pad(y, (0, pad), mode="reflect") if pad else y
        a_hat = self.branch(y)
        e1 = self.enc1(y_in)
        e2 = self.enc2(F.avg_pool1d(e1, 2))
        e3 = self.enc3(F.avg_pool1d(e2, 2))
        h = self.enc4(F.avg_pool1d(e3, 2))
        aux = {"a_hat": a_hat, "time_feat": h}
        g = self.bottleneck_proj(a_hat).unsqueeze(-1).expand(-1, -1, h.shape[-1])
        h = h + g
        d3 = self.dec3(torch.cat([self._align(self.up3(h), e3), e3], dim=1))
        d3 = self._modulate(d3, a_hat, self.film3, self.gate3, aux, "dec3")
        d2 = self.dec2(torch.cat([self._align(self.up2(d3), e2), e2], dim=1))
        d2 = self._modulate(d2, a_hat, self.film2, self.gate2, aux, "dec2")
        d1 = self.dec1(torch.cat([self._align(self.up1(d2), e1), e1], dim=1))
        d1 = self._modulate(d1, a_hat, self.film1, self.gate1, aux, "dec1")
        pred = self.out(d1)[..., :original_len]
        x_hat = y - self.residual_scale * pred if self.residual_denoising else pred
        return (x_hat, aux) if return_aux else x_hat


class PODWNO1DDenoiser(nn.Module):
    def __init__(self, rank, width=64, depth=4, levels=3, residual=True, residual_scale=1.0):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.branch = PODBranchNet(rank, width=64)
        self.stem = nn.Conv1d(1, width, 1)
        self.blocks = nn.ModuleList([MultiScaleWaveletLikeBlock(width, levels) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Linear(rank, width) for _ in range(depth)])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_aux=True):
        a_hat = self.branch(y)
        feat = self.stem(y)
        gate_values = []
        for block, gate_layer in zip(self.blocks, self.gates):
            feat = block(feat)
            gate = torch.sigmoid(gate_layer(a_hat)).unsqueeze(-1)
            feat = feat * gate
            gate_values.append(gate.detach())
        pred = self.head(feat)
        x_hat = y - self.residual_scale * pred if self.residual else pred
        aux = {"a_hat": a_hat, "gate_values": gate_values}
        return (x_hat, aux) if return_aux else x_hat


class PODTrueWNO1DDenoiser(nn.Module):
    def __init__(self, rank, width=64, depth=4, levels=3, residual=True, residual_scale=1.0):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.branch = PODBranchNet(rank, width=64)
        self.stem = nn.Conv1d(1, width, 1)
        self.blocks = nn.ModuleList([HaarWaveletOperatorBlock(width, levels) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Linear(rank, width) for _ in range(depth)])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_aux=True):
        a_hat = self.branch(y)
        feat = self.stem(y)
        gate_values = []
        for block, gate_layer in zip(self.blocks, self.gates):
            feat = block(feat)
            gate = torch.sigmoid(gate_layer(a_hat)).unsqueeze(-1)
            feat = feat * gate
            gate_values.append(gate.detach())
        pred = self.head(feat)
        x_hat = y - self.residual_scale * pred if self.residual else pred
        aux = {"a_hat": a_hat, "gate_values": gate_values}
        return (x_hat, aux) if return_aux else x_hat


class UWNO1DDenoiser(nn.Module):
    def __init__(self, base_channels=32, width=128, depth=3, levels=3, residual=False, residual_scale=1.0):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        c1, c2 = int(base_channels), int(base_channels) * 2
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, c1, 7, padding=3, bias=False),
            nn.GroupNorm(8 if c1 % 8 == 0 else 1, c1),
            nn.GELU(),
            nn.Conv1d(c1, c1, 5, padding=2, bias=False),
            nn.GroupNorm(8 if c1 % 8 == 0 else 1, c1),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(c1, c2, 5, padding=2, bias=False),
            nn.GroupNorm(8 if c2 % 8 == 0 else 1, c2),
            nn.GELU(),
            nn.Conv1d(c2, c2, 5, padding=2, bias=False),
            nn.GroupNorm(8 if c2 % 8 == 0 else 1, c2),
            nn.GELU(),
        )
        self.lift = nn.Conv1d(c2, width, 1)
        self.blocks = nn.ModuleList([HaarWaveletOperatorBlock(width, levels) for _ in range(depth)])
        self.drop = nn.Conv1d(width, c2, 1)
        self.up1 = nn.ConvTranspose1d(c2, c1, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv1d(c1 + c1, c1, 5, padding=2, bias=False),
            nn.GroupNorm(8 if c1 % 8 == 0 else 1, c1),
            nn.GELU(),
            nn.Conv1d(c1, c1, 5, padding=2, bias=False),
            nn.GroupNorm(8 if c1 % 8 == 0 else 1, c1),
            nn.GELU(),
        )
        self.head = nn.Conv1d(c1, 1, 1)

    def _align(self, x, ref):
        if x.shape[-1] == ref.shape[-1]:
            return x
        return F.interpolate(x, size=ref.shape[-1], mode="linear", align_corners=False)

    def forward(self, y, return_aux=True):
        original_len = y.shape[-1]
        pad = (2 - original_len % 2) % 2
        y_in = F.pad(y, (0, pad), mode="reflect") if pad else y
        e1 = self.enc1(y_in)
        e2 = self.enc2(F.avg_pool1d(e1, 2))
        feat = self.lift(e2)
        for block in self.blocks:
            feat = block(feat)
        b = self.drop(feat)
        d1 = self._align(self.up1(b), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        pred = self.head(d1)[..., :original_len]
        x_hat = y - self.residual_scale * pred if self.residual else pred
        aux = {}
        return (x_hat, aux) if return_aux else x_hat


class PODUWNO1DDenoiser(nn.Module):
    def __init__(
        self,
        rank,
        base_channels=32,
        width=128,
        depth=3,
        levels=3,
        residual=False,
        residual_scale=1.0,
    ):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.branch = PODBranchNet(rank, width=64)
        c1, c2 = int(base_channels), int(base_channels) * 2
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, c1, 7, padding=3, bias=False),
            nn.GroupNorm(8 if c1 % 8 == 0 else 1, c1),
            nn.GELU(),
            nn.Conv1d(c1, c1, 5, padding=2, bias=False),
            nn.GroupNorm(8 if c1 % 8 == 0 else 1, c1),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(c1, c2, 5, padding=2, bias=False),
            nn.GroupNorm(8 if c2 % 8 == 0 else 1, c2),
            nn.GELU(),
            nn.Conv1d(c2, c2, 5, padding=2, bias=False),
            nn.GroupNorm(8 if c2 % 8 == 0 else 1, c2),
            nn.GELU(),
        )
        self.lift = nn.Conv1d(c2, width, 1)
        self.blocks = nn.ModuleList([HaarWaveletOperatorBlock(width, levels) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Linear(rank, width) for _ in range(depth)])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.drop = nn.Conv1d(width, c2, 1)
        self.up1 = nn.ConvTranspose1d(c2, c1, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv1d(c1 + c1, c1, 5, padding=2, bias=False),
            nn.GroupNorm(8 if c1 % 8 == 0 else 1, c1),
            nn.GELU(),
            nn.Conv1d(c1, c1, 5, padding=2, bias=False),
            nn.GroupNorm(8 if c1 % 8 == 0 else 1, c1),
            nn.GELU(),
        )
        self.head = nn.Conv1d(c1, 1, 1)

    def _align(self, x, ref):
        if x.shape[-1] == ref.shape[-1]:
            return x
        return F.interpolate(x, size=ref.shape[-1], mode="linear", align_corners=False)

    def forward(self, y, return_aux=True):
        original_len = y.shape[-1]
        pad = (2 - original_len % 2) % 2
        y_in = F.pad(y, (0, pad), mode="reflect") if pad else y
        a_hat = self.branch(y)
        e1 = self.enc1(y_in)
        e2 = self.enc2(F.avg_pool1d(e1, 2))
        feat = self.lift(e2)
        gate_values = []
        for block, gate_layer in zip(self.blocks, self.gates):
            feat = block(feat)
            gate = torch.sigmoid(gate_layer(a_hat)).unsqueeze(-1)
            feat = feat * gate
            gate_values.append(gate.detach())
        b = self.drop(feat)
        d1 = self._align(self.up1(b), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        pred = self.head(d1)[..., :original_len]
        x_hat = y - self.residual_scale * pred if self.residual else pred
        aux = {"a_hat": a_hat, "gate_values": gate_values}
        return (x_hat, aux) if return_aux else x_hat


class ClassWisePODTrueWNO1DDenoiser(nn.Module):
    def __init__(self, num_classes, rank, width=64, depth=4, levels=3, residual=False, residual_scale=1.0):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.branch = ClassWisePODBranchNet(num_classes, rank, width=64)
        cond_dim = int(num_classes) + int(rank)
        self.stem = nn.Conv1d(1, width, 1)
        self.blocks = nn.ModuleList([HaarWaveletOperatorBlock(width, levels) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Linear(cond_dim, width) for _ in range(depth)])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_aux=True):
        class_logits, class_prob, coeff_all, coeff_mix, condition = self.branch(y)
        feat = self.stem(y)
        gate_values = []
        for block, gate_layer in zip(self.blocks, self.gates):
            feat = block(feat)
            gate = torch.sigmoid(gate_layer(condition)).unsqueeze(-1)
            feat = feat * gate
            gate_values.append(gate.detach())
        pred = self.head(feat)
        x_hat = y - self.residual_scale * pred if self.residual else pred
        aux = {
            "a_hat": coeff_mix,
            "a_hat_all": coeff_all,
            "class_logits": class_logits,
            "class_prob": class_prob,
            "gate_values": gate_values,
        }
        return (x_hat, aux) if return_aux else x_hat


class PODFNO1DDenoiser(nn.Module):
    def __init__(self, rank, width=64, modes=64, depth=4, residual=False, residual_scale=1.0):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.branch = PODBranchNet(rank, width=64)
        self.lift = nn.Conv1d(1, width, 1)
        self.blocks = nn.ModuleList([FNOBlock1D(width, modes) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Linear(rank, width) for _ in range(depth)])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.project = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_aux=True):
        a_hat = self.branch(y)
        feat = self.lift(y)
        gate_values = []
        for block, gate_layer in zip(self.blocks, self.gates):
            feat = block(feat)
            gate = torch.sigmoid(gate_layer(a_hat)).unsqueeze(-1)
            feat = feat * gate
            gate_values.append(gate.detach())
        pred = self.project(feat)
        x_hat = y - self.residual_scale * pred if self.residual else pred
        aux = {"a_hat": a_hat, "gate_values": gate_values}
        return (x_hat, aux) if return_aux else x_hat


class PODResCNN1DDenoiser(nn.Module):
    def __init__(self, rank, width=64, depth=8, kernel_size=7, residual=False, residual_scale=1.0):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.branch = PODBranchNet(rank, width=64)
        self.stem = nn.Sequential(nn.Conv1d(1, width, kernel_size, padding=kernel_size // 2), nn.GELU())
        self.blocks = nn.ModuleList([ResidualConvBlock1D(width, kernel_size=kernel_size) for _ in range(depth)])
        self.gates = nn.ModuleList([nn.Linear(rank, width) for _ in range(depth)])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_aux=True):
        a_hat = self.branch(y)
        feat = self.stem(y)
        gate_values = []
        for block, gate_layer in zip(self.blocks, self.gates):
            feat = block(feat)
            gate = torch.sigmoid(gate_layer(a_hat)).unsqueeze(-1)
            feat = feat * gate
            gate_values.append(gate.detach())
        pred = self.head(feat)
        x_hat = y - self.residual_scale * pred if self.residual else pred
        aux = {"a_hat": a_hat, "gate_values": gate_values}
        return (x_hat, aux) if return_aux else x_hat


class PODTCN1DDenoiser(nn.Module):
    def __init__(
        self,
        rank,
        width=64,
        depth=8,
        kernel_size=5,
        residual=False,
        residual_scale=1.0,
        dropout=0.05,
    ):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.branch = PODBranchNet(rank, width=64)
        self.stem = nn.Conv1d(1, width, 1)
        self.blocks = nn.ModuleList(
            [TCNBlock1D(width, kernel_size=kernel_size, dilation=2 ** (i % 6), dropout=dropout) for i in range(depth)]
        )
        self.gates = nn.ModuleList([nn.Linear(rank, width) for _ in range(depth)])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_aux=True):
        a_hat = self.branch(y)
        feat = self.stem(y)
        gate_values = []
        for block, gate_layer in zip(self.blocks, self.gates):
            feat = block(feat)
            gate = torch.sigmoid(gate_layer(a_hat)).unsqueeze(-1)
            feat = feat * gate
            gate_values.append(gate.detach())
        pred = self.head(feat)
        x_hat = y - self.residual_scale * pred if self.residual else pred
        aux = {"a_hat": a_hat, "gate_values": gate_values}
        return (x_hat, aux) if return_aux else x_hat


class PODTransformer1DDenoiser(nn.Module):
    def __init__(
        self,
        rank,
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
        self.branch = PODBranchNet(rank, width=64)
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
        self.gate = nn.Linear(rank, width)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.head = nn.Sequential(nn.Conv1d(width, width, 3, padding=1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_aux=True):
        original_len = y.shape[-1]
        pad = (self.patch_size - original_len % self.patch_size) % self.patch_size
        y_in = F.pad(y, (0, pad), mode="reflect") if pad else y
        a_hat = self.branch(y)
        tokens = self.patch(y_in).transpose(1, 2)
        tokens = self.encoder(self.pos(tokens))
        feat = self.norm(tokens).transpose(1, 2)
        gate = torch.sigmoid(self.gate(a_hat)).unsqueeze(-1)
        feat = feat * gate
        feat = F.interpolate(feat, size=y_in.shape[-1], mode="linear", align_corners=False)
        pred = self.head(feat)[..., :original_len]
        x_hat = y - self.residual_scale * pred if self.residual else pred
        aux = {"a_hat": a_hat, "gate_values": [gate.detach()]}
        return (x_hat, aux) if return_aux else x_hat


class NoPODTrueWNO1DDenoiser(nn.Module):
    def __init__(self, width=64, depth=4, levels=3, residual=True, residual_scale=1.0):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.stem = nn.Conv1d(1, width, 1)
        self.blocks = nn.ModuleList([HaarWaveletOperatorBlock(width, levels) for _ in range(depth)])
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_aux=True):
        feat = self.stem(y)
        for block in self.blocks:
            feat = block(feat)
        pred = self.head(feat)
        x_hat = y - self.residual_scale * pred if self.residual else pred
        aux = {}
        return (x_hat, aux) if return_aux else x_hat


class LightweightFrequencyAttention(nn.Module):
    def __init__(self, rank, width, max_modes=256):
        super().__init__()
        self.max_modes = int(max_modes)
        self.pod_gate = nn.Sequential(nn.Linear(rank, width), nn.Sigmoid())
        self.freq_gain = nn.Parameter(torch.zeros(1, width, self.max_modes))
        self.mix = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(8 if width % 8 == 0 else 1, width)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, a_hat):
        length = x.shape[-1]
        spec = torch.fft.rfft(x, dim=-1)
        n_freq = spec.shape[-1]
        modes = min(self.max_modes, n_freq)
        gain = torch.tanh(self.freq_gain[..., :modes])
        cond = self.pod_gate(a_hat).unsqueeze(-1)
        mult = torch.ones(spec.shape, device=spec.device, dtype=spec.real.dtype)
        low_mult = 1.0 + cond * gain
        mult = torch.cat([low_mult, mult[..., modes:]], dim=-1) if modes < n_freq else low_mult
        spec_attn = spec * mult.to(spec.dtype)
        update = torch.fft.irfft(spec_attn, n=length, dim=-1) - x
        update = self.mix(update)
        return x + self.res_scale * F.gelu(self.norm(update))


class LightweightFrequencyAttentionNoPOD(nn.Module):
    def __init__(self, width, max_modes=256):
        super().__init__()
        self.max_modes = int(max_modes)
        self.freq_gain = nn.Parameter(torch.zeros(1, width, self.max_modes))
        self.mix = nn.Conv1d(width, width, 1)
        self.norm = nn.GroupNorm(8 if width % 8 == 0 else 1, width)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        length = x.shape[-1]
        spec = torch.fft.rfft(x, dim=-1)
        n_freq = spec.shape[-1]
        modes = min(self.max_modes, n_freq)
        mult = torch.ones(spec.shape, device=spec.device, dtype=spec.real.dtype)
        low_mult = (1.0 + torch.tanh(self.freq_gain[..., :modes])).expand(spec.shape[0], -1, -1)
        mult = torch.cat([low_mult, mult[..., modes:]], dim=-1) if modes < n_freq else low_mult
        spec_attn = spec * mult.to(spec.dtype)
        update = torch.fft.irfft(spec_attn, n=length, dim=-1) - x
        update = self.mix(update)
        return x + self.res_scale * F.gelu(self.norm(update))


class PODFrequencyAttnTrueWNO1DDenoiser(nn.Module):
    def __init__(
        self,
        rank,
        width=64,
        depth=4,
        levels=3,
        residual=False,
        residual_scale=1.0,
        max_modes=256,
        attn_every_block=True,
    ):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.attn_every_block = bool(attn_every_block)
        self.branch = PODBranchNet(rank, width=64)
        self.stem = nn.Conv1d(1, width, 1)
        self.blocks = nn.ModuleList([HaarWaveletOperatorBlock(width, levels) for _ in range(depth)])
        n_attn = depth if self.attn_every_block else 1
        self.freq_attn = nn.ModuleList([LightweightFrequencyAttention(rank, width, max_modes) for _ in range(n_attn)])
        self.gates = nn.ModuleList([nn.Linear(rank, width) for _ in range(depth)])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_aux=True):
        a_hat = self.branch(y)
        feat = self.stem(y)
        gate_values = []
        for i, (block, gate_layer) in enumerate(zip(self.blocks, self.gates)):
            feat = block(feat)
            gate = torch.sigmoid(gate_layer(a_hat)).unsqueeze(-1)
            feat = feat * gate
            gate_values.append(gate.detach())
            if self.attn_every_block:
                feat = self.freq_attn[i](feat, a_hat)
        if not self.attn_every_block:
            feat = self.freq_attn[0](feat, a_hat)
        pred = self.head(feat)
        x_hat = y - self.residual_scale * pred if self.residual else pred
        aux = {"a_hat": a_hat, "gate_values": gate_values}
        return (x_hat, aux) if return_aux else x_hat


class NoPODFrequencyAttnTrueWNO1DDenoiser(nn.Module):
    def __init__(
        self,
        width=64,
        depth=4,
        levels=3,
        residual=False,
        residual_scale=1.0,
        max_modes=256,
        attn_every_block=True,
    ):
        super().__init__()
        self.residual = bool(residual)
        self.residual_scale = float(residual_scale)
        self.attn_every_block = bool(attn_every_block)
        self.stem = nn.Conv1d(1, width, 1)
        self.blocks = nn.ModuleList([HaarWaveletOperatorBlock(width, levels) for _ in range(depth)])
        n_attn = depth if self.attn_every_block else 1
        self.freq_attn = nn.ModuleList([LightweightFrequencyAttentionNoPOD(width, max_modes) for _ in range(n_attn)])
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, 1, 1))

    def forward(self, y, return_aux=True):
        feat = self.stem(y)
        for i, block in enumerate(self.blocks):
            feat = block(feat)
            if self.attn_every_block:
                feat = self.freq_attn[i](feat)
        if not self.attn_every_block:
            feat = self.freq_attn[0](feat)
        pred = self.head(feat)
        x_hat = y - self.residual_scale * pred if self.residual else pred
        aux = {}
        return (x_hat, aux) if return_aux else x_hat
