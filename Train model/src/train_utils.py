"""
Training utility functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report

class FocalLoss(nn.Module):
    """Focal Loss function"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                epochs, patience, fold, device, save_path):
    """Training function"""
    best_model_state = None
    best_f1 = -1.0
    best_acc = 0.0
    early_stop_counter = 0
    best_val_preds = []
    best_val_labels = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X.size(0)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                
                val_loss += loss.item() * X.size(0)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # Calculate metrics
        report = classification_report(val_labels, val_preds, 
                                      output_dict=True, zero_division=0)
        current_f1 = report['1']['f1-score'] if '1' in report else 0.0
        current_acc = report['accuracy']
        
        # Update best model
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_acc = current_acc
            best_model_state = model.state_dict().copy()
            best_val_preds = val_preds
            best_val_labels = val_labels
            torch.save(best_model_state, save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"[Fold {fold+1}] Early stopping at epoch {epoch+1}")
                break
        
        # Print training progress
        print(f"[Fold {fold+1}] Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val F1: {current_f1:.4f} | Val Acc: {current_acc:.4f}")
    
    return best_model_state, best_f1, best_acc, best_val_preds, best_val_labels