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

from data_loader import load_window_cache, EEGDataset
from models.fusion_model import FusionModel
from utils.metrics import FocalLoss
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
    
    # 2. Load the preprocessed window cache used by the AUG06 workflow.
    print("\n" + "="*50)
    print("1. Loading EEG windows")
    print("="*50)

    X_all, y_all, metadata_df = load_window_cache(
        config['data']['cache_npz_path'],
        config['data']['cache_metadata_path'],
    )
    groups = metadata_df['patient_id'].astype(str).to_numpy()
    print(f"Loaded {len(X_all)} cached windows from {len(np.unique(groups))} patients")
    
    # 4. 5-fold cross-validation (grouped by patient)
    print("\n" + "="*50)
    print("2. Starting 5-fold cross-validation")
    print("="*50)
    
    gkf = GroupKFold(n_splits=config['training']['n_splits'])
    
    fold_results = []
    
    splits = gkf.split(X_all, y_all, groups=groups)
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n[Fold {fold+1}/{config['training']['n_splits']}] Starting training")

        train_X = X_all[train_idx]
        train_y = y_all[train_idx]
        val_X = X_all[val_idx]
        val_y = y_all[val_idx]
        
        # Positive sample augmentation for training set
        augmentation_config = config['augmentation']
        pos_idx = np.where(train_y == 1)[0]
        if augmentation_config['enabled'] and len(pos_idx) > 0:
            augmented_X = np.array([
                augment(
                    train_X[i],
                    strategy=augmentation_config['strategy'],
                    noise_std_ratio=augmentation_config['noise_std_ratio'],
                )
                for i in pos_idx
                for _ in range(augmentation_config['repeats'])
            ])
            train_X = np.concatenate([train_X, augmented_X], axis=0)
            train_y = np.concatenate([
                train_y,
                np.ones(len(augmented_X), dtype=train_y.dtype),
            ])
        
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
            in_channels=config['model']['in_channels'],
            channel_plan=tuple(config['model']['channel_plan']),
            block_plan=tuple(config['model']['block_plan']),
            pooling=config['model']['pooling'],
            dropout_rate=config['model']['resnet_dropout'],
            spp_output_sizes=config['model']['spp_output_sizes'],
            kan_layers=config['model']['kan_layers']
        ).to(device)
        
        # Loss function and optimizer
        criterion = FocalLoss(
            alpha=config['loss']['focal_alpha'],
            gamma=config['loss']['focal_gamma'],
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
        )
        
        # Save path
        save_path = Path(config['data']['results_dir']) / f"fold_{fold+1}_best_weights.pth"
        
        # Train model
        best_model_state, best_f1, best_acc, val_preds, val_labels, val_probs = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epochs=config['training']['epochs'],
            patience=config['training']['patience'],
            min_delta=config['training']['early_stop_min_delta'],
            fold=fold,
            device=device,
            save_path=save_path
        )
        
        predictions_dir = Path(config['data']['results_dir']) / 'predictions'
        predictions_dir.mkdir(parents=True, exist_ok=True)
        prediction_rows = pd.DataFrame({
            'fold': fold + 1,
            'patient_id': groups[val_idx],
            'y_true': val_labels,
            'y_pred': val_preds,
            'prob_spike': val_probs,
        })
        for column in ('window_index', 'start_time_sec', 'mid_time_sec',
                       'end_time_sec', 'duration_sec'):
            if column in metadata_df.columns:
                prediction_rows[column] = metadata_df.iloc[val_idx][column].to_numpy()
        prediction_rows.to_csv(
            predictions_dir / f'fold_{fold + 1}_predictions.csv',
            index=False,
            encoding='utf-8-sig',
        )
        
        # Record results
        fold_results.append({
            'fold': fold + 1,
            'best_f1': best_f1,
            'best_acc': best_acc,
            'weight_path': str(save_path)
        })
        
        # Print results
        print(f"\nFold {fold+1} Results:")
        print(f"  F1 Score: {best_f1:.4f}")
        print(f"  Accuracy: {best_acc:.4f}")
    
    # 5. Summarize results
    print("\n" + "="*50)
    print("3. 5-fold Cross-validation Summary")
    print("="*50)
    
    avg_f1 = np.mean([res['best_f1'] for res in fold_results])
    avg_acc = np.mean([res['best_acc'] for res in fold_results])
    std_f1 = np.std([res['best_f1'] for res in fold_results])
    std_acc = np.std([res['best_acc'] for res in fold_results])
    
    print(f"\nSummary Results:")
    print(f"  Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"  Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    
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
