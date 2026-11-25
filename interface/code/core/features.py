
"""Feature extraction for 0.2-second multi-channel EEG windows."""
import numpy as np

def extract_sne_features(signal_1d):
    from scipy.signal import convolve
    neo = signal_1d ** 2 - np.roll(signal_1d, 1) * np.roll(signal_1d, -1)
    neo[0], neo[-1] = neo[1], neo[-2]
    tri_window = np.convolve(np.ones(9), np.ones(9), 'full')
    tri_window /= tri_window.sum()
    sne = convolve(neo, tri_window, mode='same')
    return sne


def extract_multi_channel_features(signal):
    n_channels, time_steps = signal.shape
    features = []

    for c in range(n_channels):
        x0 = signal[c]
        x1 = np.zeros_like(x0)
        x1[1:] = x0[1:] - x0[:-1]
        x2 = np.zeros_like(x0)
        x2[2:] = x1[2:] - x1[1:-1]
        sne = extract_sne_features(x0)

        features.append(np.stack([x0, x1, x2, sne], axis=0))

    return np.stack(features, axis=1)
