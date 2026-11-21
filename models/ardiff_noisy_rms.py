"""Input-only RMS adaptation for ARDiff's diffusion-coordinate noise prediction.

The backbone layers are unchanged. This is a normalization adaptation, not an
assertion that the original paper used this preprocessing or scale-equivariance.
"""
import torch
from torch import nn

from models.ardiff_reproduction import forward_sample, reverse_denoise


def sample_rms(x, eps=1e-8):
    return x.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)


def rms_diffusion_batch(clean_raw, alpha, generator=None):
    # Unit-power clean defines forward-noise synthesis units, not the input scaler.
    reference_scale = sample_rms(clean_raw)
    clean_reference = clean_raw/reference_scale
    qt, steps, noise = forward_sample(clean_reference, alpha, generator)
    state_rms = sample_rms(qt)
    a = alpha[steps-1].view(-1, 1, 1)
    # Equivalent physical additive pair: y=reference_scale/sqrt(a)*qt.
    # Thus BOTH y and clean targets use exactly the RMS of that corrupted y.
    physical_noisy_rms = reference_scale*state_rms/a.sqrt()
    return {'state': qt, 'steps': steps, 'noise': noise, 'state_rms': state_rms,
            'input_normalized': qt/state_rms,
            'clean_target_normalized': clean_raw/physical_noisy_rms,
            'noise_target_normalized': noise/state_rms,
            'physical_noisy_rms': physical_noisy_rms, 'reference_scale': reference_scale}


class NoisyRMSNoisePredictor(nn.Module):
    """Predict noise in current-state units using a normalized backbone input.

    The model sees state/RMS(state); its normalized noise prediction is multiplied
    by that same RMS before Eq. 4, maintaining the diffusion update's units.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, state, steps):
        scale = sample_rms(state)
        normalized_noise = self.backbone(state/scale, steps)
        return normalized_noise.float()*scale


class NoisyRMSStepPredictor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, noisy_state):
        return self.backbone(noisy_state/sample_rms(noisy_state))


@torch.no_grad()
def denoise_noisy_rms(model, predictor, noisy_raw, alpha, *, amp=False,
                      step_prediction_scale=1000, start_steps=None):
    """Only observed noisy input is used to normalize and restore physical units."""
    if not torch.isfinite(noisy_raw).all():
        raise ValueError('Noisy signal contains NaN/Inf')
    outer_scale = sample_rms(noisy_raw)
    restored, steps = reverse_denoise(model, predictor, noisy_raw/outer_scale, alpha,
                                     amp=amp, start_steps=start_steps,
                                     step_prediction_scale=step_prediction_scale)
    return restored*outer_scale, steps, outer_scale
