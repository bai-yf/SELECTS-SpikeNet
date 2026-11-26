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

def calculate_swi_from_times(annotation_times, predicted_positive_times,
                             total_duration_sec):
    """Calculate patient-level SWI from unique one-second epochs.

    Annotation timestamps and positive-window midpoint timestamps must use the
    same patient-relative time coordinate. Multiple events or windows within
    one integer second contribute only once.
    """
    total_seconds = int(np.ceil(float(total_duration_sec)))
    if total_seconds <= 0:
        return 0.0, 0.0, 0.0

    def occupied_seconds(times):
        values = np.asarray(times, dtype=float)
        values = values[np.isfinite(values)]
        return {
            int(np.floor(value))
            for value in values
            if 0 <= value < total_duration_sec
        }

    true_seconds = occupied_seconds(annotation_times)
    predicted_seconds = occupied_seconds(predicted_positive_times)
    swi_true = 100.0 * len(true_seconds) / total_seconds
    swi_pred = 100.0 * len(predicted_seconds) / total_seconds
    return swi_true, swi_pred, abs(swi_true - swi_pred)

def compute_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return report, cm
