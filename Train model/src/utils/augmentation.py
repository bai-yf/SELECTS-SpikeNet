"""Data augmentation used by the final AUG06 training protocol."""
import numpy as np


def add_gaussian_noise(seg, std_ratio=0.03):
    """Add small Gaussian noise without changing channel or time positions."""
    seg = np.asarray(seg)
    noise_std = std_ratio * (np.std(seg) + 1e-8)
    return seg + np.random.normal(0, noise_std, size=seg.shape)


def time_reverse(seg):
    """Reverse a ``(n_channels, time_steps)`` segment on its true time axis."""
    return np.asarray(seg)[:, ::-1].copy()


def noise_time_reverse(seg, std_ratio=0.03):
    """Apply 3% Gaussian noise and then always reverse the time axis."""
    return time_reverse(add_gaussian_noise(seg, std_ratio=std_ratio))


def augment(seg, strategy="noise_time_reverse", noise_std_ratio=0.03):
    """Dispatch the configured augmentation strategy."""
    if strategy == "none":
        return np.asarray(seg).copy()
    if strategy == "noise_time_reverse":
        return noise_time_reverse(seg, std_ratio=noise_std_ratio)
    raise ValueError(f"Unknown augmentation strategy: {strategy}")
