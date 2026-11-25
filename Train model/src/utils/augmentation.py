"""Data augmentation used by the final AUG06 training protocol."""
import numpy as np


def augment(seg):
    """Apply Gaussian noise plus true time reversal to one EEG segment.

    The complete segment is reversed along the time axis with 50% probability.
    Frequency-domain perturbation is intentionally excluded so that this code
    matches the AUG06 protocol reported in the revised experiments.
    """
    seg = np.asarray(seg).copy()

    if np.random.rand() > 0.5:
        seg = seg[:, ::-1].copy()

    seg += np.random.normal(0, 0.05 * np.std(seg), seg.shape)
    return seg