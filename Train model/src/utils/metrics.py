"""
Evaluation metrics calculation functions
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

class FocalLoss(nn.Module):
    """Binary focal loss used by the final training protocol."""
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


def compute_patient_swi_from_timestamps(
        annotation_times, window_mid_times, window_predictions,
        coverage_start_sec, coverage_end_sec):
    """Compute one patient's SWI using non-overlapping one-second epochs.

    Ground truth comes from original annotation timestamps. Predicted-positive
    0.2-second windows are assigned to seconds by ``mid_time_sec`` and combined
    with logical OR. All times must share the same coordinate system.
    """
    start_epoch = int(np.floor(float(coverage_start_sec) + 1e-9))
    stop_epoch = int(np.ceil(float(coverage_end_sec) - 1e-9))
    n_epochs = max(0, stop_epoch - start_epoch)
    if n_epochs == 0:
        raise ValueError("The prediction coverage contains no one-second epoch")

    def valid_seconds(times):
        values = np.asarray(times, dtype=float)
        values = values[np.isfinite(values)]
        values = values[
            (values >= coverage_start_sec) & (values < coverage_end_sec)
        ]
        seconds = np.floor(values).astype(int)
        return set(seconds[(seconds >= start_epoch) & (seconds < stop_epoch)])

    true_seconds = valid_seconds(annotation_times)
    mid_times = np.asarray(window_mid_times, dtype=float)
    predictions = np.asarray(window_predictions, dtype=int)
    if len(mid_times) != len(predictions):
        raise ValueError("Window midpoint and prediction arrays differ in length")
    predicted_seconds = valid_seconds(mid_times[predictions == 1])

    swi_true = 100.0 * len(true_seconds) / n_epochs
    swi_pred = 100.0 * len(predicted_seconds) / n_epochs
    return {
        'n_epochs': n_epochs,
        'true_spike_seconds': len(true_seconds),
        'predicted_spike_seconds': len(predicted_seconds),
        'swi_true': swi_true,
        'swi_pred': swi_pred,
        'swi_abs_error': abs(swi_true - swi_pred),
        'swi_method': 'annotation_1s_epoch_midtime_prediction',
    }
