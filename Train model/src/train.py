"""
Main training script - BECT EEG signal classification
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import GroupKFold

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_loader import load_patient_data, EEGDataset
from models.fusion_model import FusionModel
from utils.metrics import FocalLoss, compute_swi
from utils.augmentation import augment
from train_utils import train_model

def main():
    # 1. Load configuration
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    seed = config['training']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 2. Load data
    print("\n" + "="*50)
    print("1. Loading metadata and patient data")
    print("="*50)
    
    meta_df = pd.read_excel(config['data']['metadata_path'], header=None)
    all_patient_ids = list(set([row[0].strip()[:11] for _, row in meta_df.iterrows()]))
    print(f"Total patients: {len(all_patient_ids)}")
    
    # 3. Load EEG data
    train_val_data = load_patient_data(
        meta_df=meta_df,
        patient_ids=all_patient_ids,
        data_dir=config['data']['raw_data_dir'],
        config=config
    )
    print(f"Successfully loaded data from {len(train_val_data)} patients")
    
    # 4. 5-fold cross-validation (grouped by patient)
    print("\n" + "="*50)
    print("2. Starting 5-fold cross-validation")
    print("="*50)
    
    gkf = GroupKFold(n_splits=5)
    patient_ids_for_split = [d["pid"] for d in train_val_data]
    X_split = np.arange(len(train_val_data))
    y_split = np.zeros(len(train_val_data))
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_split, y_split, groups=patient_ids_for_split)):
        print(f"\n[Fold {fold+1}/5] Starting training")
        
        # Data preparation
        train_patient_data = [train_val_data[i] for i in train_idx]
        val_patient_data = [train_val_data[i] for i in val_idx]
        
        # Concatenate training set
        train_X = np.concatenate([d["X"] for d in train_patient_data])
        train_y = np.concatenate([d["y"] for d in train_patient_data])
        
        # Concatenate validation set
        val_X = np.concatenate([d["X"] for d in val_patient_data])
        val_y = np.concatenate([d["y"] for d in val_patient_data])
        
        # Positive sample augmentation for training set
        pos_idx = np.where(train_y == 1)[0]
        if len(pos_idx) > 0:
            augmented_X = [augment(train_X[i].T).T for i in pos_idx]
            augmented_X = np.array(augmented_X)
            train_X = np.concatenate([train_X, augmented_X], axis=0)
            train_y = np.concatenate([train_y, np.ones(len(pos_idx), dtype=train_y.dtype)], axis=0)
        
        print(f"[Fold {fold+1}] Dataset size:")
        print(f"  Training set: {train_X.shape[0]} samples (positive: {np.sum(train_y==1)}, negative: {np.sum(train_y==0)})")
        print(f"  Validation set: {val_X.shape[0]} samples (positive: {np.sum(val_y==1)}, negative: {np.sum(val_y==0)})")
        
        # Create DataLoader
        train_dataset = EEGDataset(train_X, train_y, config)
        val_dataset = EEGDataset(val_X, val_y, config)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        
        # Initialize model
        signal_config = config['signal']
        model = FusionModel(
            num_channels=signal_config['n_channels'],
            time_steps=int(signal_config['sampling_freq'] * signal_config['window_sec']),
            num_classes=2,
            spp_output_sizes=config['model']['spp_output_sizes'],
            kan_layers=config['model']['kan_layers']
        ).to(device)
        
        # Loss function and optimizer
        criterion = FocalLoss(alpha=0.65, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        
        # Save path
        save_path = Path(config['data']['results_dir']) / f"fold_{fold+1}_best_weights.pth"
        
        # Train model
        best_model_state, best_f1, best_acc, val_preds, val_labels = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=config['training']['epochs'],
            patience=config['training']['patience'],
            fold=fold,
            device=device,
            save_path=save_path
        )
        
        # Calculate SWI metrics
        swi_true, swi_pred = compute_swi(val_labels, val_preds, window_sec=signal_config['window_sec'])
        
        # Record results
        fold_results.append({
            'fold': fold + 1,
            'best_f1': best_f1,
            'best_acc': best_acc,
            'swi_true': swi_true,
            'swi_pred': swi_pred,
            'weight_path': str(save_path)
        })
        
        # Print results
        print(f"\nFold {fold+1} Results:")
        print(f"  F1 Score: {best_f1:.4f}")
        print(f"  Accuracy: {best_acc:.4f}")
        print(f"  True SWI: {swi_true:.2f}%")
        print(f"  Predicted SWI: {swi_pred:.2f}%")
        print(f"  SWI Error: {abs(swi_true - swi_pred):.2f}%")
    
    # 5. Summarize results
    print("\n" + "="*50)
    print("3. 5-fold Cross-validation Summary")
    print("="*50)
    
    avg_f1 = np.mean([res['best_f1'] for res in fold_results])
    avg_acc = np.mean([res['best_acc'] for res in fold_results])
    avg_swi_error = np.mean([abs(res['swi_true'] - res['swi_pred']) for res in fold_results])
    
    std_f1 = np.std([res['best_f1'] for res in fold_results])
    std_acc = np.std([res['best_acc'] for res in fold_results])
    std_swi_error = np.std([abs(res['swi_true'] - res['swi_pred']) for res in fold_results])
    
    print(f"\nSummary Results:")
    print(f"  Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"  Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"  Average SWI Error: {avg_swi_error:.2f}% ± {std_swi_error:.2f}%")
    
    # Save summary results to CSV
    results_df = pd.DataFrame(fold_results)
    summary_path = Path(config['data']['results_dir']) / "cross_validation_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\nDetailed results saved to: {summary_path}")
    
    # Find best fold
    best_fold_idx = np.argmax([res['best_f1'] for res in fold_results])
    best_fold = fold_results[best_fold_idx]
    print(f"\nBest Fold: Fold {best_fold['fold']}")
    print(f"  Best F1: {best_fold['best_f1']:.4f}")
    print(f"  Best Weight: {best_fold['weight_path']}")

if __name__ == "__main__":
    main()