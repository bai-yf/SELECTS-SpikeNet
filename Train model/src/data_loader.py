"""
EEG data loading and preprocessing module
"""
import os
import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


def load_window_cache(npz_path, metadata_path):
    """Load the window cache used by the final AUG06 notebook."""
    npz_path = Path(npz_path)
    metadata_path = Path(metadata_path)
    if not npz_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Window cache is incomplete: "
            f"{npz_path} and {metadata_path} are both required"
        )

    cache = np.load(npz_path, allow_pickle=True)
    X = cache["X"]
    y = cache["y"]
    metadata = pd.read_csv(metadata_path)
    if len(X) != len(y) or len(X) != len(metadata):
        raise ValueError("Window cache arrays and metadata have different lengths")
    if "patient_id" not in metadata.columns:
        raise ValueError("Window-cache metadata must contain a patient_id column")
    return X, y, metadata

def load_patient_data(meta_df, patient_ids, data_dir, config):
    """Load patient data"""
    patient_data = []
    signal_config = config['signal']
    
    for i, row in meta_df.iterrows():
        # Extract patient ID
        pid = row[0].strip()[:11]
        if pid not in patient_ids:
            continue
        
        # Load EEG file
        eeg_filename = row[0].strip() + ".set"
        filepath = Path(data_dir) / eeg_filename
        
        if not filepath.exists():
            print(f"Warning: {eeg_filename} does not exist, skipping")
            continue
        
        # Read EEG data
        raw = mne.io.read_raw_eeglab(filepath, preload=True)
        data = raw.get_data()
        
        # Segment data
        X, y = segment_eeg_data(
            data=data,
            label_times=row[1:].dropna().astype(float).tolist(),
            config=signal_config
        )
        
        if len(X) > 0:
            patient_data.append({
                'pid': pid,
                'X': np.array(X),
                'y': np.array(y)
            })
    
    return patient_data

def segment_eeg_data(data, label_times, config):
    """EEG window segmentation"""
    sfreq = config['sampling_freq']
    window_sec = config['window_sec']
    half_window = int(window_sec * sfreq / 2)
    pre_post_time = config['pre_post_time']
    
    X, y = [], []
    max_start = data.shape[1] - 2 * half_window
    
    if max_start <= 0:
        return X, y
    
    for s in range(0, max_start + 1, half_window):
        end = s + 2 * half_window
        seg = data[:, s:end]
        
        # Normalization
        seg = (seg - np.mean(seg, axis=1, keepdims=True)) / (np.std(seg, axis=1, keepdims=True) + 1e-6)
        
        # Label determination
        mid_time = (s + half_window) / sfreq
        label = 1 if any(abs(mid_time - t) <= pre_post_time for t in label_times) else 0
        
        X.append(seg)
        y.append(label)
    
    return X, y

class EEGDataset(Dataset):
    def __init__(self, X, y, config):
        self.config = config
        self.X = self.preprocess_features(X)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def preprocess_features(self, X):
        """Feature preprocessing"""
        from utils.preprocessing import extract_multi_channel_features

        features = []
        for seg in X:
            feat = extract_multi_channel_features(seg)
            features.append(feat)
        
        return torch.tensor(np.array(features), dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
