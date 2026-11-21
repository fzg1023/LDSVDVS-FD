import numpy as np


def add_awgn(signal, snr_db, rng=None, eps=1e-12):
    """Add AWGN to a single signal or a batch using the requested SNR."""
    if snr_db is None:
        return signal
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(signal, dtype=np.float32)
    axes = tuple(range(1, x.ndim)) if x.ndim > 1 else None
    power = np.mean(np.square(x), axis=axes, keepdims=True)
    noise_power = power / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32)
    noise_std = np.sqrt(np.maximum(noise_power, eps))
    return x + noise * noise_std
