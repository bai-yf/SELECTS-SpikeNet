import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mne
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

# === Environment Setup ===
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Configuration Parameters ===
data_dir = "path/to/Train model/data"  # Modify this path
meta_path = "path/to/Train model/metadata/spick_time_label.xlsx"  # Modify this path
sfreq = 500
window_sec = 0.2
half_window = int(window_sec * sfreq / 2)
pre_post_time = 0.1
batch_size = 64
epochs = 30
learning_rate = 1e-4
n_channels = 16

# === Set Random Seed ===
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# === Waveform Dictionary Parameters ===
WAVEFORM_DICT = {
    "slow_wave": {"lambdas": [2, 3, 4, 6, 8], "type": "sin"},
    "sharp_spike": {"lambdas": list(range(5, 25)), "type": "ricker"},
    "complex_wave": {"lambda_pairs": [(2,5), (3,7), (4,10), (6,15), (8,20), (2,25)], "type": "combined"},
    "haar_wave": {"lambdas": [2], "type": "haar"}
}

# === Precompute Waveform Templates ===
def precompute_waveform_templates(waveform_dict, sfreq, window_sec=0.2):
    """Precompute and save waveform templates"""
    window_samples = int(window_sec * sfreq)
    t = np.linspace(0, window_sec, window_samples, endpoint=False)
    templates = {}
    
    # 1. Slow wave templates (φ1)
    slow_templates = []
    for lambda1 in waveform_dict["slow_wave"]["lambdas"]:
        f = lambda1
        phi1 = np.zeros(window_samples)
        mask = (t >= 0) & (t < f/sfreq)
        phi1[mask] = np.sin(np.pi * t[mask] / (f/sfreq))
        wave = np.sqrt(lambda1) * (phi1 - np.mean(phi1))
        wave /= np.std(wave) if np.std(wave) != 0 else 1.0
        slow_templates.append(wave)
    templates["slow_wave"] = np.array(slow_templates)  # shape: (n_lambdas, n_samples)
    
    # 2. Sharp wave templates (φ2)
    sharp_templates = []
    for lambda2 in waveform_dict["sharp_spike"]["lambdas"]:
        f = lambda2
        phi2 = np.zeros(window_samples)
        mask1 = (t >= 0) & (t < 0.4*f/sfreq)
        mask2 = (t >= 0.4*f/sfreq) & (t < f/sfreq)
        phi2[mask1] = 5 * t[mask1] / (2 * f/sfreq)
        phi2[mask2] = 5 * t[mask2] / (3 * f/sfreq) + 5/3
        wave = np.sqrt(lambda2) * (phi2 - np.mean(phi2))
        wave /= np.std(wave) if np.std(wave) != 0 else 1.0
        sharp_templates.append(wave)
    templates["sharp_spike"] = np.array(sharp_templates)
    
    # 3. Complex wave templates (φ3)
    complex_templates = []
    for lambda13, lambda23 in waveform_dict["complex_wave"]["lambda_pairs"]:
        f1, f2 = lambda13, lambda23
        phi1 = np.zeros(window_samples)
        mask1 = (t >= 0) & (t < f1/sfreq)
        phi1[mask1] = np.sin(np.pi * t[mask1] / (f1/sfreq))
        
        phi2 = np.zeros(window_samples)
        mask2_1 = (t >= 0) & (t < 0.4*f2/sfreq)
        mask2_2 = (t >= 0.4*f2/sfreq) & (t < f2/sfreq)
        phi2[mask2_1] = 5 * t[mask2_1] / (2 * f2/sfreq)
        phi2[mask2_2] = 5 * t[mask2_2] / (3 * f2/sfreq) + 5/3
        
        phi3 = phi1 + phi2
        wave = (phi3 - np.mean(phi3)) / (np.std(phi3) if np.std(phi3) != 0 else 1.0)
        complex_templates.append(wave)
    templates["complex_wave"] = np.array(complex_templates)
    
    # 4. Haar wave templates (φ4)
    haar_templates = []
    for lambda4 in waveform_dict["haar_wave"]["lambdas"]:
        f = lambda4
        phi4 = np.zeros(window_samples)
        mask_haar = (t >= 0) & (t < f/sfreq)
        phi4[mask_haar] = np.where(t[mask_haar] < (f/sfreq)/2, 1, -1)
        wave = np.sqrt(lambda4) * (phi4 - np.mean(phi4))
        wave /= np.std(wave) if np.std(wave) != 0 else 1.0
        haar_templates.append(wave)
    templates["haar_wave"] = np.array(haar_templates)
    
    # Save templates to file
    save_waveform_templates(templates)
    print(f"Successfully precomputed and saved waveform templates")
    
    return templates

# Save/Load templates
def save_waveform_templates(templates, path="path/to/save/waveform_templates.pkl"):
    """Save waveform templates to file"""
    with open(path, "wb") as f:
        pickle.dump(templates, f)
    print(f"Waveform templates saved to {path}")

def load_waveform_templates(path="path/to/save/waveform_templates.pkl"):
    """Load waveform templates from file"""
    if os.path.exists(path):
        with open(path, "rb") as f:
            print(f"Loading waveform templates from {path}")
            return pickle.load(f)
    else:
        print("Waveform template file not found, recomputing...")
        return precompute_waveform_templates(WAVEFORM_DICT, sfreq)

# Preload templates
WAVEFORM_TEMPLATES = load_waveform_templates()

# === GPU-accelerated Feature Extraction ===
def extract_waveform_features_gpu(eeg_segment, templates=WAVEFORM_TEMPLATES):
    """Extract waveform features using GPU acceleration"""
    n_channels, n_samples = eeg_segment.shape
    
    # Convert input to PyTorch tensor and move to GPU
    eeg_tensor = torch.tensor(eeg_segment, dtype=torch.float32, device=device).unsqueeze(1)  # (n_channels, 1, n_samples)
    
    # Initialize features for each waveform type
    slow_features = []
    sharp_features = []
    complex_features = []
    haar_features = []
    
    # Calculate slow wave features
    slow_templates = torch.tensor(templates["slow_wave"], dtype=torch.float32, device=device)  # shape: (n_lambdas, n_samples)
    for template in slow_templates:
        template = template.unsqueeze(0).unsqueeze(1)  # shape: (1, 1, n_samples)
        conv = torch.nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=template.shape[-1],
            padding='same',
            bias=False
        ).to(device)
        conv.weight.data = template
        with torch.no_grad():
            feature = conv(eeg_tensor)  # shape: (n_channels, 1, n_samples)
        slow_features.append(feature)
        # Free unused GPU memory
        torch.cuda.empty_cache()
    slow_features = torch.cat(slow_features, dim=1).mean(dim=1).unsqueeze(-1)  # shape: (n_channels, 1, n_samples)
    # Free unused GPU memory
    torch.cuda.empty_cache()
    
    # Calculate sharp wave features
    sharp_templates = torch.tensor(templates["sharp_spike"], dtype=torch.float32, device=device)  # shape: (n_lambdas, n_samples)
    for template in sharp_templates:
        template = template.unsqueeze(0).unsqueeze(1)  # shape: (1, 1, n_samples)
        conv = torch.nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=template.shape[-1],
            padding='same',
            bias=False
        ).to(device)
        conv.weight.data = template
        with torch.no_grad():
            feature = conv(eeg_tensor)  # shape: (n_channels, 1, n_samples)
        sharp_features.append(feature)
        # Free unused GPU memory
        torch.cuda.empty_cache()
    sharp_features = torch.cat(sharp_features, dim=1).mean(dim=1).unsqueeze(-1)  # shape: (n_channels, 1, n_samples)
    # Free unused GPU memory
    torch.cuda.empty_cache()
    
    # Calculate complex wave features
    complex_templates = torch.tensor(templates["complex_wave"], dtype=torch.float32, device=device)  # shape: (n_pairs, n_samples)
    for template in complex_templates:
        template = template.unsqueeze(0).unsqueeze(1)  # shape: (1, 1, n_samples)
        conv = torch.nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=template.shape[-1],
            padding='same',
            bias=False
        ).to(device)
        conv.weight.data = template
        with torch.no_grad():
            feature = conv(eeg_tensor)  # shape: (n_channels, 1, n_samples)
        complex_features.append(feature)
        # Free unused GPU memory
        torch.cuda.empty_cache()
    complex_features = torch.cat(complex_features, dim=1).mean(dim=1).unsqueeze(-1)  # shape: (n_channels, 1, n_samples)
    # Free unused GPU memory
    torch.cuda.empty_cache()
    
    # Calculate Haar wave features
    haar_templates = torch.tensor(templates["haar_wave"], dtype=torch.float32, device=device)  # shape: (n_lambdas, n_samples)
    for template in haar_templates:
        template = template.unsqueeze(0).unsqueeze(1)  # shape: (1, 1, n_samples)
        conv = torch.nn.Conv1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=template.shape[-1],
            padding='same',
            bias=False
        ).to(device)
        conv.weight.data = template
        with torch.no_grad():
            feature = conv(eeg_tensor)  # shape: (n_channels, 1, n_samples)
        haar_features.append(feature)
        # Free unused GPU memory
        torch.cuda.empty_cache()
    haar_features = torch.cat(haar_features, dim=1).mean(dim=1).unsqueeze(-1)  # shape: (n_channels, 1, n_samples)
    # Free unused GPU memory
    torch.cuda.empty_cache()
    
    # Combine all features
    all_features = torch.cat([slow_features, sharp_features, complex_features, haar_features], dim=2)  # shape: (n_channels, n_samples, 4)
    return all_features.cpu().numpy()

# === Dataset Class ===
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].permute(0, 2, 1).to(device), self.y[idx]  # x (C, F, T)
    
# === Data Loading ===
def load_patient_data(meta_df, patient_ids, data_dir):
    """Load patient EEG data"""
    patient_data = []
    for i, row in meta_df.iterrows():
        pid = row[0].strip()[:11]
        if pid not in patient_ids:
            continue
            
        filepath = os.path.join(data_dir, f"{row[0].strip()}.set")
        label_times = row[1:].dropna().astype(float).tolist()
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} does not exist, skipping")
            continue
            
        raw = mne.io.read_raw_eeglab(filepath, preload=True)
        data = raw.get_data()
        data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
        
        # Extract segments
        X, y = [], []
        max_start = data.shape[1] - half_window * 2
        
        for s in range(0, max_start, half_window):
            mid_time = (s + half_window) / sfreq
            label = 1 if any(abs(mid_time - t) <= pre_post_time for t in label_times) else 0
            seg = data[:, s:s+half_window*2]
            wave_features = extract_waveform_features_gpu(seg)
            combined = np.concatenate([seg[..., np.newaxis], wave_features], axis=-1)
            X.append(combined)
            y.append(label)
        
        if len(X) > 0:
            patient_data.append({
                'pid': pid,
                'X': np.array(X),
                'y': np.array(y)
            })
    
    return patient_data

# === Model Definition ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, n_channels=16, n_features=5, d_model=80, nhead=4, num_layers=1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=half_window*2)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        # x shape: (batch_size, channels=16, features=5, time=100) → reshape to (time, batch, channels*features)
        b, c, f, t = x.shape
        x = x.permute(3, 0, 2, 1).reshape(t, b, f*c)  # (100, b, 80)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=0)  # (b, 80)
        return self.fc(x)

# === Loss Function ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return loss.mean()

# === SWI Calculation ===
def compute_swi(y_true, y_pred, window_sec):
    """Calculate Spike Wave Index (SWI)"""
    samples_per_sec = int(1/window_sec)
    sec_true = [int(any(y_true[i:i+samples_per_sec])) 
               for i in range(0, len(y_true), samples_per_sec)]
    sec_pred = [int(any(y_pred[i:i+samples_per_sec])) 
               for i in range(0, len(y_pred), samples_per_sec)]
    swi_true = np.mean(sec_true) * 100
    swi_pred = np.mean(sec_pred) * 100
    return swi_true, swi_pred

# === Training Function ===
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience=5):
    """Train model with early stopping"""
    best_model = None
    best_f1 = -1
    early_stop_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            class_output = model(X)
            
            # Calculate loss
            loss = criterion(class_output, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                val_loss += criterion(outputs, y).item()
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        # Print training progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        
        # Calculate F1 score
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        val_f1 = report['1']['f1-score'] if '1' in report else 0
        
        # Early stopping and model saving
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = copy.deepcopy(model)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return best_model

# === Main Execution ===
if __name__ == "__main__":
    # 1. Load metadata and split patients
    meta_df = pd.read_excel(meta_path, header=None)
    all_patient_ids = list(set([row[0].strip()[:11] for _, row in meta_df.iterrows()]))
    train_val_patient_ids = [pid for pid in all_patient_ids]
    
    print(f"Training and validation patients: {train_val_patient_ids}")

    # 2. Load training and validation data
    train_val_data = load_patient_data(meta_df, train_val_patient_ids, data_dir)
    
    # 3. 5-fold cross-validation
    gkf = GroupKFold(n_splits=5)
    patient_ids = [d['pid'] for d in train_val_data]
    X = np.arange(len(train_val_data))
    y = np.zeros(len(train_val_data))  # dummy labels
    
    best_models = []
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=patient_ids)):
        print(f"\n=== Fold {fold+1} ===")
        
        # Prepare data
        train_data = [train_val_data[i] for i in train_idx]
        val_data = [train_val_data[i] for i in val_idx]
        
        train_X = np.concatenate([d['X'] for d in train_data])
        train_y = np.concatenate([d['y'] for d in train_data])
        val_X = np.concatenate([d['X'] for d in val_data])
        val_y = np.concatenate([d['y'] for d in val_data])
        
        # Print dimension check
        print("Original X shape:", train_X.shape)
        print("Original y shape:", train_y.shape)

        # Create DataLoader
        train_loader = DataLoader(EEGDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(EEGDataset(val_X, val_y), batch_size=batch_size)
        
        # Initialize model
        model = TransformerClassifier().to(device)
        
        # Loss function and optimizer
        criterion = FocalLoss(alpha=0.65, gamma=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train model
        best_model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
        
        # Validation set evaluation
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = best_model(X)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        
        # Calculate SWI
        swi_true, swi_pred = compute_swi(val_labels, val_preds, window_sec)
        
        # Save results
        report = classification_report(val_labels, val_preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(val_labels, val_preds)
        fold_results.append({
            'fold': fold,
            'report': report,
            'confusion_matrix': cm,
            'swi_true': swi_true,
            'swi_pred': swi_pred,
            'model': best_model
        })
        
        # Print results
        print(f"Fold {fold+1} Validation Results:")
        print(pd.DataFrame(report).transpose().round(3))
        print("Confusion Matrix:")
        print(cm)
        print(f"SWI - True: {swi_true:.2f}%, Predicted: {swi_pred:.2f}%, Error: {abs(swi_true-swi_pred):.2f}%")
    
    # 4. Summary of all folds
    print("\n" + "="*50)
    print("Cross-validation Summary")
    print("="*50)
    
    avg_f1 = np.mean([res['report']['1']['f1-score'] if '1' in res['report'] else 0 for res in fold_results])
    avg_acc = np.mean([res['report']['accuracy'] for res in fold_results])
    avg_swi_error = np.mean([abs(res['swi_true'] - res['swi_pred']) for res in fold_results])
    
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average SWI Error: {avg_swi_error:.2f}%")
    
    # Save models
    for i, result in enumerate(fold_results):
        model_path = f"fold_{i+1}_best_model.pth"
        torch.save(result['model'].state_dict(), model_path)
        print(f"Model saved to {model_path}")