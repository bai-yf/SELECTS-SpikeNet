"""
Data augmentation functions
"""
import numpy as np

def augment(seg):
    """Augment a single EEG segment"""
    seg = seg.copy()
    
    # 1. Time reversal (50% probability)
    if np.random.rand() > 0.5:
        seg = seg[:, ::-1]
    
    # 2. Frequency perturbation (50% probability)
    if np.random.rand() > 0.5:
        fft = np.fft.rfft(seg, axis=1)
        fft *= np.random.uniform(0.8, 1.2, fft.shape)
        seg = np.fft.irfft(fft, axis=1)
    
    # 3. Additive Gaussian noise
    seg += np.random.normal(0, 0.05 * np.std(seg), seg.shape)
    
    return seg