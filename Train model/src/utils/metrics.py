"""
Evaluation metrics calculation functions
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

class FocalLoss(nn.Module):
    """Binary focal loss used by the final AUG06 notebook."""
    def __init__(self, alpha=0.65, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(ce_loss, self.alpha),
            torch.full_like(ce_loss, 1.0 - self.alpha),
        )
        return (alpha_t * (1 - pt) ** self.gamma * ce_loss).mean()

def compute_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return report, cm
