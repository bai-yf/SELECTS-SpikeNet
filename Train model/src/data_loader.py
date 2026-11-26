"""Cached EEG-window loading and feature preparation."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


def load_window_cache(npz_path, metadata_path):
    """Load the preprocessed window cache and its metadata."""
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
