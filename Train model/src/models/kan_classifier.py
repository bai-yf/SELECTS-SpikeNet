"""
KAN Classifier Module
"""
import sys
import torch
import torch.nn as nn

# Add KAN library path (modify based on actual installation path)
# sys.path.append('/path/to/efficient-kan/src')  # Modify to your KAN library path
from efficient_kan.kan import KAN

class KANClassifier(nn.Module):
    """KAN Classifier"""
    def __init__(self, feature_dim, num_classes, hidden_layers=[128, 64]):
        super().__init__()
        # Build KAN layer structure
        kan_layers = [feature_dim] + list(hidden_layers) + [num_classes]
        self.kan = KAN(kan_layers)
    
    def forward(self, x):
        return self.kan(x)
