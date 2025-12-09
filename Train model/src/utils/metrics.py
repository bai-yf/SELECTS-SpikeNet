"""
Evaluation metrics calculation functions
"""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class FocalLoss:
    """Focal Loss function"""
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, inputs, targets):
        import torch
        import torch.nn.functional as F
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def compute_swi(y_true, y_pred, window_sec):
    """Calculate SWI metric"""
    samples_per_sec = int(1 / window_sec)
    
    # Aggregate by second
    sec_true = []
    sec_pred = []
    
    for i in range(0, len(y_true), samples_per_sec):
        end_idx = min(i + samples_per_sec, len(y_true))
        sec_true.append(int(any(y_true[i:end_idx])))
        sec_pred.append(int(any(y_pred[i:end_idx])))
    
    # Calculate percentage
    swi_true = np.mean(sec_true) * 100
    swi_pred = np.mean(sec_pred) * 100
    
    return swi_true, swi_pred

def compute_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return report, cm