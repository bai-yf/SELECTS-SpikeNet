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

# === Environment Setup ===
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
n_channels = 16  # Note: Should match actual data channel count, new model original in_channels=19, may need alignment
test_size = 0.2  # Independent test set proportion

# === Set Random Seed ===
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# === Data Augmentation ===
def augment(seg):
    """Augment EEG segment"""
    if np.random.rand() > 0.5:
        seg = seg[:, ::-1]  # Time reversal
    
    fft = np.fft.rfft(seg, axis=1)
    if np.random.rand() > 0.5:
        fft *= np.random.uniform(0.8, 1.2, fft.shape)
    seg = np.fft.irfft(fft, axis=1)
    seg += np.random.normal(0, 0.05, seg.shape)
    return seg

# === Dataset Class ===
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === Data Loading ===
def load_patient_data(meta_df, patient_ids, data_dir):
    """Load and preprocess patient EEG data"""
    patient_data = []
    for i, row in meta_df.iterrows():
        pid = row[0].strip()[:11]  # Extract patient ID
        if pid not in patient_ids:
            continue
            
        filepath = os.path.join(data_dir, f"{row[0].strip()}.set")
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found, skipping")
            continue
            
        label_times = row[1:].dropna().astype(float).tolist()
        
        try:
            raw = mne.io.read_raw_eeglab(filepath, preload=True)
        except Exception as e:
            print(f"Error reading file {filepath}: {str(e)}, skipping")
            continue
            
        data = raw.get_data()
        # Channel-wise normalization
        data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)
        
        # Extract segments
        X, y = [], []
        max_start = data.shape[1] - half_window * 2
        
        if max_start <= 0:
            print(f"Warning: Data too short in file {filepath}, skipping")
            continue
            
        for s in range(0, max_start, half_window):
            mid_time = (s + half_window) / sfreq
            label = 1 if any(abs(mid_time - t) <= pre_post_time for t in label_times) else 0
            seg = data[:, s:s+half_window*2].T  # (time, channels)
            X.append(seg)
            y.append(label)
        
        if X:  # Ensure samples exist
            patient_data.append({
                'pid': pid,
                'X': np.array(X),
                'y': np.array(y)
            })
    return patient_data

# === New Model Dependency Modules (direct copy of your SeizureTransformer related classes) ===
class SpatialDropout1d(nn.Module):
    """1D Spatial Dropout"""
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate)
        
    def forward(self, x):
        x = x.unsqueeze(dim=-1)  # Add fake dimension
        x = self.dropout(x)
        x = x.squeeze(dim=-1)  # Remove fake dimension
        return x

class ResCNNBlock(nn.Module):
    """Residual CNN Block"""
    def __init__(self, filters, ker, drop_rate):
        super().__init__()
        self.manual_padding = False
        if ker == 3:
            padding = 1
        else:
            self.manual_padding = True
            padding = 0
        self.dropout = SpatialDropout1d(drop_rate)
        self.norm1 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv1 = nn.Conv1d(filters, filters, ker, padding=padding)
        self.norm2 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv2 = nn.Conv1d(filters, filters, ker, padding=padding)
        
    def forward(self, x):
        y = self.norm1(x)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv1(y)
        y = self.norm2(y)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv2(y)
        return x + y

class ResCNNStack(nn.Module):
    """Stack of Residual CNN Blocks"""
    def __init__(self, kernel_sizes, filters, drop_rate):
        super().__init__()
        members = []
        for ker in kernel_sizes:
            members.append(ResCNNBlock(filters, ker, drop_rate))
        self.members = nn.ModuleList(members)
        
    def forward(self, x):
        for member in self.members:
            x = member(x)
        return x

class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Encoder(nn.Module):
    """Encoder for local feature extraction"""
    def __init__(self, input_channels, filters, kernel_sizes, in_samples):
        super().__init__()
        convs = []
        pools = []
        elus = []
        self.paddings = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )
            padding = in_samples % 2
            self.paddings.append(padding)
            pools.append(nn.MaxPool1d(2, padding=0))
            elus.append(nn.ELU(inplace=True))
            in_samples = (in_samples + padding) // 2
        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)
        self.elus = nn.ModuleList(elus)
        
    def forward(self, x):
        skips = []
        for conv, pool, padding, elu in zip(self.convs, self.pools, self.paddings, self.elus):
            x = elu(conv(x))
            skips.append(x)
            if padding != 0:
                x = F.pad(x, (0, padding), "constant", -1e10)
            x = pool(x)
        return x, skips

class SeizureTransformer(nn.Module):
    """Seizure Transformer Model for EEG Classification"""
    def __init__(
        self,
        in_channels=16,
        drop_rate=0.1,
        num_classes=2  # Explicit number of classification classes
    ):
        super().__init__()
        self.in_channels = in_channels
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.filters = [32, 64, 128, 256, 512]
        self.kernel_sizes = [11, 9, 7, 7, 5, 5, 3]
        self.res_cnn_kernels = [3, 3, 3, 3, 2, 3, 2]

        # Encoder (extracts local features)
        self.encoder = Encoder(
            input_channels=self.in_channels,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=100  # Can initialize based on actual input segment length
        )

        # ResCNN+Transformer (enhances features and captures long-range dependencies)
        self.res_cnn_stack = ResCNNStack(
            kernel_sizes=self.res_cnn_kernels,
            filters=self.filters[-1],
            drop_rate=self.drop_rate,
        )
        self.position_encoding = PositionalEncoding(d_model=512)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=4,
            dim_feedforward=2048
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=8
        )

        # Classification head (outputs classes directly based on global features)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling to compress time dimension
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Input shape: (B, T, C) → (B, C, T)
        x = x.permute(0, 2, 1)
        
        # Encoder feature extraction
        x, _ = self.encoder(x)  # Ignore skip connections (decoder not needed)
        
        # ResCNN+Transformer enhancement
        x = self.res_cnn_stack(x)
        x = x.permute(2, 0, 1)  # (T, B, C)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)  # (B, C, T)
        
        # Classification output
        out = self.classifier(x)
        return out

# === Focal Loss ===
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
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
    """Calculate Spike Wave Index"""
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
            loss = criterion(model(X), y)
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
    
    print(f"Training/validation patients: {train_val_patient_ids}")
    
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
        
        # Data augmentation
        # Get positive class sample indices
        pos_idx = np.where(train_y == 1)[0]

        # Augment each positive sample
        augmented_X = [augment(train_X[i].T).T for i in pos_idx]
        augmented_X = np.array(augmented_X)  # shape: (len(pos_idx), C, T)

        # Dimension check
        print("Original X shape:", train_X.shape)
        print("Augmented X shape:", augmented_X.shape)

        # Concatenate data
        train_X = np.concatenate([train_X, augmented_X], axis=0)
        train_y = np.concatenate([train_y, np.ones(len(pos_idx), dtype=train_y.dtype)], axis=0)
                
        # Create DataLoader
        train_loader = DataLoader(EEGDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(EEGDataset(val_X, val_y), batch_size=batch_size)
        
        # Train model
        model = SeizureTransformer(in_channels=n_channels).to(device)
        criterion = FocalLoss(alpha=0.65, gamma=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        best_model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
        
        # Validation set evaluation
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = best_model(X)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        
        # Save results
        report = classification_report(val_labels, val_preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(val_labels, val_preds)
        fold_results.append({
            'fold': fold,
            'report': report,
            'confusion_matrix': cm,
            'model': best_model
        })
        
        print(f"Fold {fold+1} Validation Results:")
        print(pd.DataFrame(report).transpose().round(3))
        print("Confusion Matrix:")
        print(cm)
    
    # 4. Summarize results
    print("\n" + "="*60)
    print("Cross-Validation Summary")
    print("="*60)
    
    # Calculate average F1 scores across folds
    avg_f1 = np.mean([res['report']['1']['f1-score'] if '1' in res['report'] else 0 for res in fold_results])
    avg_acc = np.mean([res['report']['accuracy'] for res in fold_results])
    
    print(f"Average F1 Score across folds: {avg_f1:.4f}")
    print(f"Average Accuracy across folds: {avg_acc:.4f}")
    
    # Find best fold
    best_fold_idx = np.argmax([res['report']['1']['f1-score'] if '1' in res['report'] else 0 for res in fold_results])
    best_fold = fold_results[best_fold_idx]
    print(f"\nBest fold: Fold {best_fold['fold'] + 1}")
    print(f"Best F1 Score: {best_fold['report']['1']['f1-score']:.4f}")

# === Additional Utility Functions (if needed in future) ===
def save_model(model, path):
    """Save model to file"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path, in_channels=16):
    """Load model from file"""
    model = model_class(in_channels=in_channels)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model

# === Instructions for Path Configuration ===
"""
1. Modify line 28: data_dir = "path/to/eeg/data"
2. Modify line 29: meta_path = "path/to/metadata/file.xlsx"
3. Ensure the following file structure:
   - EEG data directory containing .set files
   - Metadata Excel file with patient information and spike annotations
4. Run the script: python seizure_transformer.py
"""