"""Adapt the existing direct-output U-Net to the shared denoiser interface."""

from .denoiser_unet import DenoiserUNet1D


class UNet1DBaseline(DenoiserUNet1D):
    def __init__(self, base_channels=32):
        super().__init__(base_channels=base_channels, mode="direct")

    def forward(self, y, return_aux=False, return_residual=False):
        pred = super().forward(y)
        if return_aux:
            return pred, {}
        return (pred, y - pred) if return_residual else pred
