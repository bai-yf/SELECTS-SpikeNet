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
from scipy.signal import hilbert
from scipy.stats import pearsonr
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import GroupKFold
from torch_geometric.nn import GATConv
from torch_geometric.data import Batch, Data

# === Environment Setup ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Configuration Parameters ===
data_dir = "path/to/Train model/data"  # Modify this path
meta_path = "path/to/Train model/metadata/spick_time_label.xlsx"  # Modify this path
sfreq = 500
window_sec = 0.2
half_window = int(window_sec * sfreq / 2)  # 50 (0.1 second * 500Hz)
pre_post_time = 0.1
batch_size = 64
epochs = 30
learning_rate = 1e-4
n_channels = 16  # 16 EEG channels
test_size = 0.2  # Independent test set proportion
top_k = 3  # Number of edges to keep per node
graph_method = 'PLV'  # 'correlation' or 'PLV'

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
    # seg input shape: (C, T), ensure output also (C, T)
    if np.random.rand() > 0.5:
        seg = seg[:, ::-1]  # Time reversal
    
    fft = np.fft.rfft(seg, axis=1)
    if np.random.rand() > 0.5:
        fft *= np.random.uniform(0.8, 1.2, fft.shape)
    seg = np.fft.irfft(fft, n=seg.shape[1], axis=1)  # Ensure time length unchanged
    
    seg += np.random.normal(0, 0.05, seg.shape)
    return seg

# === Graph Structure Related Functions ===
def keep_topk(adj_mat, top_k, directed=True):
    """Keep only top-k strongest connections for each node"""
    n = adj_mat.shape[0]
    adj_topk = np.zeros_like(adj_mat)
    for i in range(n):
        top_indices = adj_mat[i].argsort()[-top_k:][::-1]
        adj_topk[i, top_indices] = adj_mat[i, top_indices]
    if not directed:
        adj_topk = (adj_topk + adj_topk.T) / 2
        np.fill_diagonal(adj_topk, 0)
    return adj_topk

# === Dataset Class (using PyTorch Geometric's Batch for graph processing) ===
class EEGDataset(Dataset):
    def __init__(self, X, y, sensor_ids=None, graph_method='correlation', top_k=3):
        self.X = torch.tensor(X, dtype=torch.float32)  # [num_samples, num_channels, time_points]
        self.y = torch.tensor(y, dtype=torch.long)
        self.sensor_ids = sensor_ids if sensor_ids is not None else list(range(X.shape[1]))
        self.graph_method = graph_method
        self.top_k = top_k
        self.graphs = self._precompute_graphs()

    def _precompute_graphs(self):
        """Precompute graph structures for all samples"""
        graphs = []
        for i in range(len(self.X)):
            seg = self.X[i].numpy()  # [C, T]
            edge_index, edge_weight = self._get_indiv_graphs(seg)
            graphs.append((edge_index, edge_weight))
        return graphs

    def _get_indiv_graphs(self, eeg_clip):
        """Compute individual graph for each EEG clip"""
        num_sensors = len(self.sensor_ids)
        adj_mat = np.eye(num_sensors, dtype=np.float32)
        eeg_clip = eeg_clip.reshape(num_sensors, -1)
        
        # Compute connectivity matrix
        for i in range(num_sensors):
            for j in range(i + 1, num_sensors):
                if self.graph_method == 'correlation':
                    value = self.calc_correlation(eeg_clip[i], eeg_clip[j])
                elif self.graph_method == 'PLV':
                    value = self.calc_PLV(eeg_clip[i], eeg_clip[j])
                else:
                    raise ValueError(f"Invalid method: {self.graph_method}")
                adj_mat[i, j] = value
                adj_mat[j, i] = value
        
        # Keep only top-k connections
        adj_mat = keep_topk(adj_mat, top_k=self.top_k, directed=True)
        
        # Convert to edge format
        edge_index = np.array(np.nonzero(adj_mat), dtype=np.int64)
        edge_weight = adj_mat[edge_index[0], edge_index[1]]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        return edge_index, edge_weight

    def calc_correlation(self, sig1, sig2):
        """Calculate Pearson correlation coefficient"""
        corr, _ = pearsonr(sig1, sig2)
        return np.abs(corr)

    def calc_PLV(self, sig1, sig2):
        """Calculate Phase Locking Value"""
        phase1 = np.angle(hilbert(sig1))
        phase2 = np.angle(hilbert(sig2))
        phase_diff = phase1 - phase2
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        return plv

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """Return EEG data, label, edge indices, and edge weights for a single sample"""
        return self.X[idx], self.y[idx], self.graphs[idx][0], self.graphs[idx][1]

# === Custom Batch Processing Function (solve graph index concatenation) ===
def collate_fn(batch):
    """Combine multiple sample graph data into batches, automatically shift node indices"""
    X_list, y_list, edge_index_list, edge_weight_list = zip(*batch)
    
    # Concatenate EEG data and labels
    X = torch.stack(X_list, dim=0)  # [B, C, T]
    y = torch.stack(y_list, dim=0)  # [B]
    
    # Build PyTorch Geometric Data object list
    data_list = []
    for x, label, edge_idx, edge_w in zip(X_list, y_list, edge_index_list, edge_weight_list):
        # Each sample's node features: [C, T] → num_nodes=C, feature_dim=T
        node_features = x  # [C, T]
        data = Data(
            x=node_features,  # [C, T]
            edge_index=edge_idx,  # [2, E]
            edge_attr=edge_w,  # [E]
            y=label  # [1]
        )
        data_list.append(data)
    
    # Use PyTorch Geometric's Batch to automatically handle node index shifting
    batch_data = Batch.from_data_list(data_list)
    return X, y, batch_data.edge_index, batch_data.edge_attr

# === Model Definition (corrected GAT input processing) ===
class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.dropout(x)

class GNNLayer(nn.Module):
    """Graph Attention Network Layer"""
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=0.2)
        self.bn = nn.BatchNorm1d(out_channels * heads)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.bn(x)
        x = self.relu(x)
        return self.dropout(x)

class GAT_TCN(nn.Module):
    """GAT-TCN Fusion Model for EEG Classification"""
    def __init__(self, num_channels, time_steps, num_classes):
        super().__init__()
        # GAT part (spatial feature extraction)
        self.gnn1 = GNNLayer(in_channels=time_steps, out_channels=32)  # Input feature dim=T
        self.gnn2 = GNNLayer(in_channels=32*4, out_channels=64)  # 32*4=128 (multi-head concatenation)
        
        # TCN part (temporal feature extraction)
        self.tcn_blocks = nn.Sequential(
            TCNBlock(in_channels=num_channels, out_channels=64, kernel_size=3),
            TCNBlock(in_channels=64, out_channels=128, kernel_size=3, dilation=2),
            TCNBlock(in_channels=128, out_channels=256, kernel_size=3, dilation=4)
        )
        
        # Classification head
        self.fc = nn.Linear(256 + 256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        # x: (B, C, T) → EEG time-series data
        B, C, T = x.size()
        
        # === GAT Processing (Spatial Feature Extraction) ===
        # 1. Extract each sample's node features: convert batch data to [B*C, T]
        # After Batch processing, node features are automatically shifted by sample
        gat_input = x.flatten(0, 1)  # (B*C, T) → corresponds to batch-shifted node indices
        
        # 2. GAT feature extraction
        gat_out = self.gnn1(gat_input, edge_index)  # (B*C, 32*4=128)
        gat_out = self.gnn2(gat_out, edge_index)    # (B*C, 64*4=256)
        
        # 3. Aggregate spatial features for each sample
        gat_out = gat_out.view(B, C, -1)  # (B, C, 256)
        gat_aggregated = torch.mean(gat_out, dim=1)  # (B, 256) → spatial features per sample
        
        # === TCN Processing (Temporal Feature Extraction) ===
        tcn_out = self.tcn_blocks(x)  # (B, 256, T)
        tcn_aggregated = torch.mean(tcn_out, dim=2)  # (B, 256) → temporal features per sample
        
        # === Feature Fusion and Classification ===
        combined = torch.cat([gat_aggregated, tcn_aggregated], dim=1)  # (B, 512)
        combined = self.dropout(combined)
        return self.fc(combined)

# === Focal Loss ===
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
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
        for X, y, edge_index, edge_weight in train_loader:
            X, y, edge_index = X.to(device), y.to(device), edge_index.to(device)
            optimizer.zero_grad()
            outputs = model(X, edge_index)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y, edge_index, edge_weight in val_loader:
                X, y, edge_index = X.to(device), y.to(device), edge_index.to(device)
                outputs = model(X, edge_index)
                val_loss += criterion(outputs, y).item()
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        # Calculate F1 score
        report = classification_report(all_labels, all_preds, output_dict=True)
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

# === Data Loading ===
def load_patient_data(meta_df, patient_ids, data_dir):
    """Load and preprocess patient EEG data"""
    patient_data = []
    for i, row in meta_df.iterrows():
        pid = row[0].strip()[:11]
        if pid not in patient_ids:
            continue
            
        filepath = os.path.join(data_dir, f"{row[0].strip()}.set")
        label_times = row[1:].dropna().astype(float).tolist()
        
        try:
            raw = mne.io.read_raw_eeglab(filepath, preload=True)
        except:
            print(f"Skipping file {filepath} (read error)")
            continue
            
        data = raw.get_data()
        # Channel-wise normalization
        data = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)
        
        X, y = [], []
        # Sliding window to extract samples
        max_start = data.shape[1] - half_window * 2
        if max_start <= 0:
            print(f"Warning: File {filepath} data too short, skipping")
            continue
            
        for s in range(0, max_start + 1, half_window):
            mid_time = (s + half_window) / sfreq
            label = 1 if any(abs(mid_time - t) <= pre_post_time for t in label_times) else 0
            seg = data[:, s:s+half_window*2]  # (n_channels, 100)
            X.append(seg)
            y.append(label)
        
        if X:  # Ensure there are samples
            patient_data.append({
                'pid': pid,
                'X': np.array(X),  # (num_samples, n_channels, 100)
                'y': np.array(y)   # (num_samples,)
            })
    return patient_data

# === Main Execution ===
if __name__ == "__main__":
    # Load metadata and split patients
    meta_df = pd.read_excel(meta_path, header=None)
    all_patient_ids = list(set([row[0].strip()[:11] for _, row in meta_df.iterrows()]))
    
    # Split independent test set
    test_patient_ids = random.sample(all_patient_ids, int(len(all_patient_ids)*test_size))
    train_val_patient_ids = [pid for pid in all_patient_ids if pid not in test_patient_ids]
    
    print(f"Training/validation patients: {train_val_patient_ids}")
    print(f"Independent test patients: {test_patient_ids}")

    # Load training/validation data
    train_val_data = load_patient_data(meta_df, train_val_patient_ids, data_dir)
    
    # 5-fold cross-validation
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
        
        # Concatenate training data
        train_X = np.concatenate([d['X'] for d in train_data], axis=0)  # (total_samples, C, T)
        train_y = np.concatenate([d['y'] for d in train_data], axis=0)  # (total_samples,)
        
        # Data augmentation
        pos_idx = np.where(train_y == 1)[0]
        if len(pos_idx) > 0:
            augmented_X = [augment(train_X[i]) for i in pos_idx]
            augmented_X = np.array(augmented_X)
            train_X = np.concatenate([train_X, augmented_X], axis=0)
            train_y = np.concatenate([train_y, np.ones(len(pos_idx), dtype=train_y.dtype)], axis=0)
        
        # Concatenate validation data
        val_X = np.concatenate([d['X'] for d in val_data], axis=0)
        val_y = np.concatenate([d['y'] for d in val_data], axis=0)
                
        # Create DataLoader (using custom collate_fn for graph batch processing)
        train_loader = DataLoader(
            EEGDataset(train_X, train_y, graph_method=graph_method, top_k=top_k),
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=collate_fn  # Key modification: use custom batch processing
        )
        val_loader = DataLoader(
            EEGDataset(val_X, val_y, graph_method=graph_method, top_k=top_k),
            batch_size=batch_size,
            collate_fn=collate_fn  # Key modification: use custom batch processing
        )
        
        # Initialize model
        model = GAT_TCN(
            num_channels=n_channels,
            time_steps=half_window*2,
            num_classes=2
        ).to(device)
        
        # Define loss function and optimizer
        criterion = FocalLoss(alpha=0.65, gamma=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train model
        best_model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
        
        # Validation set evaluation
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch, edge_index, _ in val_loader:
                X_batch, y_batch, edge_index = X_batch.to(device), y_batch.to(device), edge_index.to(device)
                outputs = best_model(X_batch, edge_index)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        # Save results
        report = classification_report(val_labels, val_preds, output_dict=True)
        cm = confusion_matrix(val_labels, val_preds)
        kappa = cohen_kappa_score(val_labels, val_preds)
        fold_results.append({
            'fold': fold,
            'report': report,
            'confusion_matrix': cm,
            'kappa': kappa,
            'model': best_model
        })
        
        print(f"Fold {fold+1} Validation Results:")
        print(pd.DataFrame(report).transpose().round(3))
        print("Confusion Matrix:")
        print(cm)
        print(f"Fold {fold+1} Kappa Coefficient: {kappa:.4f}")

    # Select best model (based on validation F1)
    best_fold = max(enumerate(fold_results), key=lambda x: x[1]['report']['1']['f1-score'])[0]
    best_model = fold_results[best_fold]['model']
    print(f"\nSelected model from Fold {best_fold+1} (Validation F1={fold_results[best_fold]['report']['1']['f1-score']:.3f})")

    # Evaluate on independent test set
    test_data = load_patient_data(meta_df, test_patient_ids, data_dir)
    print("\n=== Independent Test Set Results ===")
    
    for patient in test_data:
        pid = patient['pid']
        X_test = patient['X']
        y_test = patient['y']
        
        # Create test set DataLoader
        test_dataset = EEGDataset(X_test, y_test, graph_method=graph_method, top_k=top_k)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            collate_fn=collate_fn  # Key modification: use custom batch processing
        )
        
        # Prediction
        preds, labels = [], []
        with torch.no_grad():
            for X_batch, y_batch, edge_index, _ in test_loader:
                X_batch, y_batch, edge_index = X_batch.to(device), y_batch.to(device), edge_index.to(device)
                outputs = best_model(X_batch, edge_index)
                preds.extend(outputs.argmax(1).cpu().numpy())
                labels.extend(y_batch.cpu().numpy())
        
        # Calculate metrics
        report = classification_report(labels, preds, output_dict=True)
        cm = confusion_matrix(labels, preds)
        swi_true, swi_pred = compute_swi(labels, preds, window_sec)
        
        print(f"\nPatient {pid} Test Results:")
        print(pd.DataFrame(report).transpose().round(3))
        print("Confusion Matrix:")
        print(cm)
        kappa_test = cohen_kappa_score(labels, preds)
        print(f"Kappa Coefficient: {kappa_test:.4f}")
        print(f"SWI: True={swi_true:.1f}% Predicted={swi_pred:.1f}%")