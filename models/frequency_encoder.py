import torch
from torch import nn


class FrequencyEncoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_channels=None, dropout=0.1):
        super().__init__()
        hidden_channels = hidden_channels or [32, 64, 128]
        layers = []
        in_ch = 1
        for out_ch in hidden_channels:
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, latent_dim)

    def forward(self, x):
        feat = self.backbone(x)
        pooled = self.pool(feat).squeeze(-1)
        return self.head(pooled)
