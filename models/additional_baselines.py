"""Additional 1-D denoising baselines using direct clean-signal prediction."""
import torch
from torch import nn
import torch.nn.functional as F
from mambapy.mamba import Mamba, MambaConfig


class DnCNN1D(nn.Module):
    def __init__(self, width=64, depth=17):
        super().__init__()
        layers = [nn.Conv1d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.extend([nn.Conv1d(width, width, 3, padding=1, bias=False),
                           nn.BatchNorm1d(width), nn.ReLU(inplace=True)])
        layers.append(nn.Conv1d(width, 1, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, return_aux=False):
        out = self.net(x)
        return (out, {}) if return_aux else out


class PatchMamba1D(nn.Module):
    def __init__(self, width=64, depth=3, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed = nn.Conv1d(1, width, patch_size, stride=patch_size)
        self.encoder = Mamba(MambaConfig(d_model=width, n_layers=depth, d_state=16, use_cuda=False))
        self.norm = nn.LayerNorm(width)
        self.decode = nn.Linear(width, patch_size)

    def forward(self, x, return_aux=False):
        length = x.shape[-1]
        pad = (-length) % self.patch_size
        z = F.pad(x, (0, pad)) if pad else x
        z = self.embed(z).transpose(1, 2)
        z = self.decode(self.norm(self.encoder(z)))
        out = z.reshape(z.shape[0], 1, -1)[..., :length]
        return (out, {}) if return_aux else out
