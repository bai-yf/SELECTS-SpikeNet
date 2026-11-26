# Training and evaluation code exported from eegpt_pretrained_encoder_head_16ch_0p2s_finetune.ipynb
# Configure paths in CONFIG before running.

# %% [notebook cell 1]
import os
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parents[1]
# ======================================================================================
# SOTA baseline：EEGPT 预训练 encoder 接分类头下游微调（16 通道 0.2 秒适配）
# 实验编号：SOTA_EEGPT_PRETRAINED_ENCODER_HEAD_16CH_0P2S_FINETUNE
# 说明：加载本地官方 EEGPT checkpoint；保持相同患者划分、缓存窗口、增强、训练数据和 Focal Loss。
# 原始代码来源：/home/baiyifei/BECT/无单独测试/整理版_可追溯实验_20260809/notebooks/00_main_lightresnet34_spp_kan.ipynb
#
# 使用方式：运行这一个 cell 即可完成当前实验的 5 折训练、评估和表格保存。
# 重要：旧 t-SNE 代码已移除。后续 t-SNE/PR 曲线/ROC 曲线请单独写可视化 notebook。
# ======================================================================================

# ===== 1. 实验配置 =====
CONFIG = {
    'experiment_id': 'SOTA_EEGPT_PRETRAINED_ENCODER_HEAD_16CH_0P2S_FINETUNE',
    'title': 'SOTA baseline：EEGPT 预训练 encoder 接分类头下游微调（16 通道 0.2 秒适配）',
    'notes': '官方 EEGPT 预训练 encoder 下游微调；保持与主实验相同患者级划分、缓存窗口、正样本增强、训练样本和 Focal Loss。EEGPT 预训练输入为 raw EEG，因此本 baseline 使用缓存中的标准化 raw window，并将 0.2s/500Hz 输入插值到 256 点以适配预训练 patch encoder；下游分类头直接接 encoder 表示，不使用预训练重建器输出。',
    'eegpt_repo_path': os.environ.get('EEGPT_REPO_PATH', str(MODEL_DIR / 'upstream' / 'EEGPT' / 'downstream')),
    'pretrained_checkpoint_path': os.environ.get('EEGPT_PRETRAINED_CHECKPOINT', str(MODEL_DIR / 'upstream' / 'EEGPT' / 'downstream' / 'checkpoint' / 'eegpt_mcae_58chs_4s_large4E.ckpt')),
    'reference_paper': 'Wang et al., EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals, NeurIPS 2024',
    'reference_repo': 'https://github.com/BINE022/EEGPT',
    'data_dir': os.environ.get('SELECTS_DATA_DIR', str(REPO_ROOT / 'data' / '44person_3-70')),
    'meta_path': os.environ.get('SELECTS_META_PATH', str(REPO_ROOT / 'data' / '44person.xlsx')),
    'result_dir': os.environ.get('SELECTS_RESULT_DIR', str(MODEL_DIR / 'outputs')),
    'cache_npz_path': os.environ.get('SELECTS_CACHE_NPZ', str(REPO_ROOT / 'cache' / 'dataset_windows_0p2s_500hz_44person.npz')),
    'cache_metadata_path': os.environ.get('SELECTS_CACHE_METADATA', str(REPO_ROOT / 'cache' / 'dataset_windows_0p2s_500hz_44person_metadata.csv')),
    'use_window_cache': True,
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
    'seed': 42,
    'sfreq': 500,
    'window_sec': 0.2,
    'half_window': int(0.2 * 500 / 2),
    'n_channels': 16,
    'in_channels': 1,
    'num_classes': 2,
    'batch_size': 64,
    'epochs': 50,
    'patience': 5,
    'early_stop_min_delta': 0.003,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'n_splits': 5,
    'augment_positive': True,
    'augmentation_strategy': 'noise_time_reverse',
    'augmentation_repeats': 1,
    'noise_std_ratio': 0.03,
    'channel_names': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6'],
    'eegpt_channel_names': ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8'],
    'eegpt_channel_name_note': 'T3/T4/T5/T6 are mapped to modern 10-10 aliases T7/T8/P7/P8 for EEGPT channel embeddings; raw channel order is unchanged.',
    'aug_vis_n_samples': 500,
    'focal_alpha': 0.65,
    'focal_gamma': 2.0,
    'backbone': 'EEGPT_pretrained_mcae_58ch_4s_large4E_encoder_head_adapted_16ch_0p2s',
    'head': 'encoder_mean_pool_linear_constraint',
    'eegpt_img_time_len': 256,
    'eegpt_patch_size': 64,
    'eegpt_patch_stride': 32,
    'eegpt_use_mean_pooling': True,
    'eegpt_use_chan_conv': False,
    'eegpt_freeze_encoder': False,
    'eegpt_freeze_reconstructor': True,
    'eegpt_loaded_pretrained_tensors': None,
    'pretraining': 'uses local official EEGPT checkpoint eegpt_mcae_58chs_4s_large4E.ckpt pretrained on mixed EEG data (58 channels, 256 Hz, 4 s, patch size 64); downstream fine-tuned on this study training folds',
    'tuning_policy': 'Pretrained fine-tuning adaptation: use standardized raw EEG windows from the shared cache, map the 16-channel montage to EEGPT channel embeddings, and interpolate each 0.2s/500Hz window from 100 to 256 samples inside EEGPTEncoderHeadClassifier. With patch_size=64 and patch_stride=32, the adapted input yields 7 temporal patches. This is input-length adaptation for the pretrained encoder, not physiological resampling or extension to the original 4s pretraining window. The downstream classifier uses target_encoder representations with temporal-patch and summary-token mean pooling plus a task-specific linear-constrained head; the reconstruction module is not used in the classification forward pass. Encoder/head are fine-tuned with the same FocalLoss and training loop. No test-fold hyperparameter search is performed.',
    'fair_comparison_constraints': ['same GroupKFold patient split', 'same cached 0.2s z-scored raw windows', 'same positive-sample augmentation', 'same train/validation windows', 'same FocalLoss alpha/gamma', 'same optimizer, learning rate, weight decay, epochs, patience, and early-stop metric'],
    'uses_raw_only_due_to_pretraining': True,
    'excluded_old_tsne_code': True,
}

# 必须在 import torch 之前屏蔽坏卡；物理 GPU 4 会在 PyTorch 中显示为 cuda:0。
os.environ['CUDA_VISIBLE_DEVICES'] = CONFIG['cuda_visible_devices']
print('CUDA_VISIBLE_DEVICES =', os.environ.get('CUDA_VISIBLE_DEVICES'))

# ===== 2. 导入依赖 =====
import copy
import json
import random
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, CONFIG['eegpt_repo_path'])
from Modules.models.EEGPT_mcae_finetune import EEGPTClassifier


class EEGPTEncoderHeadClassifier(EEGPTClassifier):
    """EEGPT downstream variant: classify from pretrained encoder tokens only."""

    def forward_features(self, x, chan_ids=None, return_patch_tokens=False, return_all_tokens=False, **kwargs):
        if chan_ids is None:
            chan_ids = self.chans_id
        if self.use_chan_conv:
            x = self.chan_conv(x)
        x = self.target_encoder(x, chan_ids.to(x))  # [B, temporal_patches, embed_tokens, 512]
        x = self.norm(x)
        if self.fc_norm is not None:
            x = self.fc_norm(x)
        if return_all_tokens:
            return x
        if return_patch_tokens:
            return x.flatten(1, 2)
        return x.mean(dim=(1, 2))


device = torch.device("cuda:0")
print(f'使用设备：{device}')

# ===== 3. 随机种子、特征和数据集 =====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_gaussian_noise(seg, std_ratio=0.03):
    """小幅高斯噪声：模拟采集噪声，不改变通道顺序和时间位置。"""
    noise_std = std_ratio * (np.std(seg) + 1e-8)
    return seg + np.random.normal(0, noise_std, size=seg.shape)


def time_reverse(seg):
    """沿 ``(n_channels, time_steps)`` 的真实时间轴反转。"""
    return np.asarray(seg)[:, ::-1].copy()


def noise_time_reverse(seg, std_ratio=0.03):
    """先加 3% 高斯噪声，再始终沿真实时间轴反转。"""
    return time_reverse(add_gaussian_noise(seg, std_ratio=std_ratio))


def augment(seg, config):
    """Apply the configured transform to one positive training window."""
    strategy = config['augmentation_strategy']
    if strategy == 'none':
        return np.asarray(seg).copy()
    if strategy == 'noise_time_reverse':
        return noise_time_reverse(seg, std_ratio=config['noise_std_ratio'])
    raise ValueError(f"Unknown augmentation strategy: {strategy}")

def extract_sne_features(signal_1d):
    """平滑非线性能量特征 SNE。"""
    from scipy.signal import convolve
    neo = signal_1d ** 2 - np.roll(signal_1d, 1) * np.roll(signal_1d, -1)
    neo[0], neo[-1] = neo[1], neo[-2]
    tri_window = np.convolve(np.ones(9), np.ones(9), 'full')
    tri_window /= tri_window.sum()
    return convolve(neo, tri_window, mode='same')


def extract_multi_channel_features(signal):
    """每个 EEG 片段转为 4 个输入特征：raw、一阶差分、二阶差分、SNE。"""
    n_channels, _ = signal.shape
    features = []
    for c in range(n_channels):
        x0 = signal[c]
        x1 = np.zeros_like(x0)
        x1[1:] = x0[1:] - x0[:-1]
        x2 = np.zeros_like(x0)
        x2[2:] = x1[2:] - x1[1:-1]
        sne = extract_sne_features(x0)
        features.append(np.stack([x0, x1, x2, sne], axis=0))
    return np.stack(features, axis=1)


class EEGDataset(Dataset):
    def __init__(self, X, y):
        # EEGPT was pretrained on raw EEG. Use the z-scored raw windows from the shared cache.
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




def load_window_cache(config):
    """读取预处理好的窗口缓存，避免每个实验重复读取 .set 和重新滑窗。"""
    cache_npz_path = Path(config['cache_npz_path'])
    cache_metadata_path = Path(config['cache_metadata_path'])
    if not cache_npz_path.exists() or not cache_metadata_path.exists():
        raise FileNotFoundError(
            '未找到窗口缓存。请先运行 notebooks/00_build_window_cache.ipynb。\n'
            f'缺失文件：{cache_npz_path} 或 {cache_metadata_path}'
        )
    cache = np.load(cache_npz_path, allow_pickle=True)
    X_all = cache['X']
    y_all = cache['y']
    metadata_df = pd.read_csv(cache_metadata_path)
    if len(X_all) != len(y_all) or len(X_all) != len(metadata_df):
        raise ValueError('缓存数据长度不一致，请重新运行 00_build_window_cache.ipynb。')
    print(f'已读取缓存：{cache_npz_path}')
    print(f'窗口数：{len(y_all)}；正类：{int(np.sum(y_all == 1))}；负类：{int(np.sum(y_all == 0))}；正类比例：{np.mean(y_all == 1):.4f}')
    return X_all, y_all, metadata_df



# ===== 4. 模型定义：EEGPT 预训练权重下游微调（16 通道 0.2 秒适配） =====
class EEGPTFineTuneModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = EEGPTEncoderHeadClassifier(
            num_classes=config['num_classes'],
            in_channels=config['n_channels'],
            img_size=[config['n_channels'], config['eegpt_img_time_len']],
            patch_stride=config['eegpt_patch_stride'],
            use_channels_names=config['eegpt_channel_names'],
            use_chan_conv=config['eegpt_use_chan_conv'],
            use_mean_pooling=config['eegpt_use_mean_pooling'],
            use_freeze_encoder=config['eegpt_freeze_encoder'],
            use_freeze_reconstructor=config['eegpt_freeze_reconstructor'],
            desired_time_len=config['eegpt_img_time_len'],
        )
        load_report = self._load_pretrained(config['pretrained_checkpoint_path'])
        if hasattr(self.backbone, 'reconstructor'):
            for p in self.backbone.reconstructor.parameters():
                p.requires_grad_(False)
        self.pretrained_load_report = load_report
        self.feature_map_shape = ('EEGPT encoder temporal+summary mean pooled token',)
        self.feature_dim = self.backbone.embed_dim

    def _load_pretrained(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'未找到 EEGPT 预训练权重：{checkpoint_path}')
        try:
            ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        except TypeError:
            ckpt = torch.load(str(checkpoint_path), map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        model_state = self.backbone.state_dict()
        mapped_state = {}
        skipped_shape = []
        for key, value in state_dict.items():
            new_key = None
            if key.startswith('encoder.'):
                new_key = 'target_encoder.' + key[len('encoder.'):]
            elif key.startswith('target_encoder.'):
                new_key = key
            if new_key and new_key in model_state:
                if model_state[new_key].shape == value.shape:
                    mapped_state[new_key] = value
                else:
                    skipped_shape.append((new_key, tuple(value.shape), tuple(model_state[new_key].shape)))
        missing, unexpected = self.backbone.load_state_dict(mapped_state, strict=False)
        print(f'EEGPT 预训练权重加载：matched={len(mapped_state)} missing={len(missing)} unexpected={len(unexpected)} skipped_shape={len(skipped_shape)}')
        if missing:
            print('缺失/未加载参数示例（含未使用的重建器和新分类头）：', missing[:8])
        if skipped_shape:
            print('形状不匹配跳过示例：', skipped_shape[:5])
        return {
            'checkpoint_path': str(checkpoint_path),
            'matched_tensors': len(mapped_state),
            'missing_keys': list(missing),
            'unexpected_keys': list(unexpected),
            'skipped_shape_mismatch': skipped_shape,
        }

    def forward(self, x):
        return self.backbone(x)


class FusionModel(EEGPTFineTuneModel):
    pass


class FocalLoss(nn.Module):
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

# ===== 5. 指标计算：审稿人要求的指标全部从 TP/FP/TN/FN 和概率重算 =====
def safe_div(num, den):
    return float(num / den) if den else 0.0


def compute_weighted_specificity(y_true, y_pred):
    """WS：Weighted Specificity。按每类 support 加权的 one-vs-rest specificity。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = [0, 1]
    total = len(y_true)
    values = []
    for c in classes:
        support = int(np.sum(y_true == c))
        y_true_c = (y_true == c).astype(int)
        y_pred_c = (y_pred == c).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_c, y_pred_c, labels=[0, 1]).ravel()
        specificity_c = safe_div(tn, tn + fp)
        values.append((support / total) * specificity_c if total else 0.0)
    return float(np.sum(values))


def compute_all_metrics(y_true, y_pred, y_prob, window_sec):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average='weighted', zero_division=0)
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)

    try:
        pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        pr_auc = np.nan
    try:
        roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        roc_auc = np.nan
    duration_min = len(y_true) * window_sec / 60
    duration_hour = len(y_true) * window_sec / 3600

    return {
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
        'support_non_spike': int(support[0]), 'support_spike': int(support[1]),
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_precision': macro_p, 'macro_recall': macro_r, 'macro_f1': macro_f1,
        'weighted_precision': weighted_p, 'weighted_recall': weighted_r, 'weighted_f1': weighted_f1,
        'weighted_specificity_WS': compute_weighted_specificity(y_true, y_pred),
        'non_spike_precision': per_class_p[0], 'non_spike_recall': per_class_r[0], 'non_spike_f1': per_class_f1[0],
        'spike_precision': per_class_p[1], 'spike_recall_sensitivity': per_class_r[1], 'spike_f1': per_class_f1[1],
        'specificity': safe_div(tn, tn + fp),
        'fpr': safe_div(fp, fp + tn),
        'fnr': safe_div(fn, fn + tp),
        'fdr': safe_div(fp, fp + tp),
        'npv': safe_div(tn, tn + fn),
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'fp_per_min': safe_div(fp, duration_min),
        'fp_per_hour': safe_div(fp, duration_hour),
    }


def format_mean_std(mean_value, std_value, digits=4):
    if pd.isna(mean_value):
        return 'NA'
    if pd.isna(std_value):
        return f'{mean_value:.{digits}f}'
    return f'{mean_value:.{digits}f} ± {std_value:.{digits}f}'


def build_summary_tables(fold_summary_df, config):
    metric_cols = [c for c in fold_summary_df.columns if c not in ['fold', 'weight_path']]
    rows = []
    for metric in metric_cols:
        if pd.api.types.is_numeric_dtype(fold_summary_df[metric]):
            rows.append({
                'experiment_id': config['experiment_id'],
                'metric': metric,
                'mean': fold_summary_df[metric].mean(),
                'std': fold_summary_df[metric].std(ddof=1),
                'mean±std': format_mean_std(fold_summary_df[metric].mean(), fold_summary_df[metric].std(ddof=1)),
            })
    aggregate_df = pd.DataFrame(rows)

    main_metrics = ['accuracy', 'macro_f1', 'spike_precision', 'spike_recall_sensitivity', 'spike_f1', 'specificity', 'fpr', 'pr_auc']
    paper_row = {'experiment_id': config['experiment_id'], 'model': config['title']}
    for metric in main_metrics:
        sub = aggregate_df[aggregate_df['metric'] == metric]
        paper_row[metric] = sub['mean±std'].iloc[0] if len(sub) else 'NA'
    paper_main_df = pd.DataFrame([paper_row])

    reviewer_metrics = [
        'accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
        'weighted_precision', 'weighted_recall', 'weighted_f1', 'weighted_specificity_WS',
        'spike_precision', 'spike_recall_sensitivity', 'spike_f1',
        'specificity', 'fpr', 'fdr', 'fp_per_min', 'fp_per_hour',
        'pr_auc', 'roc_auc',
    ]
    reviewer_df = aggregate_df[aggregate_df['metric'].isin(reviewer_metrics)].copy()
    reviewer_df['metric'] = pd.Categorical(reviewer_df['metric'], categories=reviewer_metrics, ordered=True)
    reviewer_df = reviewer_df.sort_values('metric')
    return aggregate_df, paper_main_df, reviewer_df


def save_all_tables(result_dir, config, fold_summary_df, aggregate_df, paper_main_df, reviewer_df):
    result_dir = Path(result_dir)
    fold_summary_df.to_csv(result_dir / 'supplement_per_fold_table.csv', index=False, encoding='utf-8-sig')
    aggregate_df.to_csv(result_dir / 'aggregate_summary.csv', index=False, encoding='utf-8-sig')
    paper_main_df.to_csv(result_dir / 'paper_main_table.csv', index=False, encoding='utf-8-sig')
    reviewer_df.to_csv(result_dir / 'reviewer_metrics_table.csv', index=False, encoding='utf-8-sig')

    metric_definitions = """# 指标说明\n\n- TP/FP/TN/FN：以 spike 为正类。\n- Specificity = TN / (TN + FP)。\n- FPR = FP / (FP + TN)。\n- FDR = FP / (TP + FP)。\n- Spike Recall/Sensitivity = TP / (TP + FN)。\n- PR-AUC：使用 spike 类预测概率计算。\n- WS：Weighted Specificity，不是 Weighted Sensitivity；按各类别 support 加权的 one-vs-rest specificity。\n- paper_main_table.csv：正文表推荐使用。\n- supplement_per_fold_table.csv：补充材料逐折结果。\n- reviewer_metrics_table.csv：给审稿人说明用的完整均值±标准差指标表。\n- predictions/fold_x_predictions.csv：逐窗口原始预测，可重算所有指标。\n"""
    (result_dir / 'metric_definitions.md').write_text(metric_definitions, encoding='utf-8')

    try:
        with pd.ExcelWriter(result_dir / 'all_tables.xlsx') as writer:
            paper_main_df.to_excel(writer, sheet_name='正文表', index=False)
            fold_summary_df.to_excel(writer, sheet_name='逐折补充表', index=False)
            reviewer_df.to_excel(writer, sheet_name='审稿人指标表', index=False)
            aggregate_df.to_excel(writer, sheet_name='全部均值标准差', index=False)
    except Exception as exc:
        print(f'提示：Excel 文件未生成，仅保存 CSV。原因：{exc}')


# ===== 6. 增强样本可视化与生理范围质控 =====
def collect_positive_windows(patient_data, max_samples, seed=42):
    rng = np.random.default_rng(seed)
    positives = []
    for item in patient_data:
        pos_idx = np.where(item['y'] == 1)[0]
        for idx in pos_idx:
            positives.append(item['X'][idx])
    if len(positives) == 0:
        return np.empty((0, CONFIG['n_channels'], 2 * CONFIG['half_window']))
    positives = np.array(positives)
    if len(positives) > max_samples:
        keep = rng.choice(len(positives), size=max_samples, replace=False)
        positives = positives[keep]
    return positives


def signal_quality_features(samples):
    """提取简单生理范围统计，用于比较增强前后是否明显越界。"""
    rows = []
    for sample in samples:
        diff_t = np.diff(sample, axis=1)
        rows.append({
            'mean': float(np.mean(sample)),
            'std': float(np.std(sample)),
            'abs_max': float(np.max(np.abs(sample))),
            'ptp': float(np.ptp(sample)),
            'rms': float(np.sqrt(np.mean(sample ** 2))),
            'line_length': float(np.mean(np.sum(np.abs(diff_t), axis=1))),
        })
    return pd.DataFrame(rows)


def save_embedding_plot(points, labels, title, save_path):
    plt.figure(figsize=(7, 6))
    colors = {'original': '#2f6fbb', 'augmented': '#d95f02'}
    for group in ['original', 'augmented']:
        mask = labels == group
        plt.scatter(points[mask, 0], points[mask, 1], s=14, alpha=0.65, label=group, c=colors[group])
    plt.title(title)
    plt.xlabel('dim-1')
    plt.ylabel('dim-2')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_augmentation_quality_from_cache(X_all, y_all, config, result_dir):
    """保存增强前/增强后正样本的 t-SNE/UMAP 图和统计表。该步骤只做质控，不参与训练。"""
    vis_dir = Path(result_dir) / 'augmentation_visualization'
    vis_dir.mkdir(parents=True, exist_ok=True)
    strategy = config['augmentation_strategy']

    pos_idx = np.where(y_all == 1)[0]
    if len(pos_idx) == 0:
        (vis_dir / 'README.txt').write_text('没有可视化样本：未找到正样本。', encoding='utf-8')
        return
    rng = np.random.default_rng(config['seed'])
    if len(pos_idx) > config['aug_vis_n_samples']:
        pos_idx = rng.choice(pos_idx, size=config['aug_vis_n_samples'], replace=False)
    original = X_all[pos_idx]
    if len(original) == 0:
        (vis_dir / 'README.txt').write_text('没有可视化样本：未找到正样本。', encoding='utf-8')
        return

    if strategy == 'none':
        original_stats = signal_quality_features(original)
        original_stats['group'] = 'original'
        original_stats.to_csv(vis_dir / 'original_only_signal_stats.csv', index=False, encoding='utf-8-sig')
        (vis_dir / 'README.txt').write_text('当前实验为不增强 baseline，因此没有增强后样本可视化。', encoding='utf-8')
        return

    augmented = np.array([augment(x, config) for x in original])
    original_stats = signal_quality_features(original)
    augmented_stats = signal_quality_features(augmented)
    original_stats['group'] = 'original'
    augmented_stats['group'] = 'augmented'
    pd.concat([original_stats, augmented_stats], ignore_index=True).to_csv(vis_dir / 'augmentation_signal_stats.csv', index=False, encoding='utf-8-sig')

    pair_rows = []
    for i, (orig, aug) in enumerate(zip(original, augmented)):
        orig_flat = orig.reshape(-1)
        aug_flat = aug.reshape(-1)
        corr = np.corrcoef(orig_flat, aug_flat)[0, 1]
        pair_rows.append({
            'sample_index': i,
            'corr_original_augmented': float(corr),
            'rms_ratio_augmented_over_original': float((np.sqrt(np.mean(aug ** 2)) + 1e-8) / (np.sqrt(np.mean(orig ** 2)) + 1e-8)),
            'abs_max_ratio_augmented_over_original': float((np.max(np.abs(aug)) + 1e-8) / (np.max(np.abs(orig)) + 1e-8)),
            'mean_abs_difference': float(np.mean(np.abs(aug - orig))),
        })
    pd.DataFrame(pair_rows).to_csv(vis_dir / 'augmentation_pairwise_quality.csv', index=False, encoding='utf-8-sig')

    X_vis = np.concatenate([original, augmented], axis=0).reshape(len(original) * 2, -1)
    labels = np.array(['original'] * len(original) + ['augmented'] * len(augmented))
    X_vis = (X_vis - X_vis.mean(axis=0, keepdims=True)) / (X_vis.std(axis=0, keepdims=True) + 1e-8)
    n_components = min(30, X_vis.shape[0] - 1, X_vis.shape[1])
    X_pca = PCA(n_components=n_components, random_state=config['seed']).fit_transform(X_vis)

    perplexity = max(5, min(30, (X_pca.shape[0] - 1) // 3))
    tsne_points = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=config['seed']).fit_transform(X_pca)
    save_embedding_plot(tsne_points, labels, f"t-SNE: 原始正样本 vs 增强正样本 ({strategy})", vis_dir / 'tsne_original_vs_augmented.png')
    pd.DataFrame({'group': labels, 'x': tsne_points[:, 0], 'y': tsne_points[:, 1]}).to_csv(vis_dir / 'tsne_points.csv', index=False, encoding='utf-8-sig')

    try:
        import umap
        umap_points = umap.UMAP(n_components=2, random_state=config['seed']).fit_transform(X_pca)
        save_embedding_plot(umap_points, labels, f"UMAP: 原始正样本 vs 增强正样本 ({strategy})", vis_dir / 'umap_original_vs_augmented.png')
        pd.DataFrame({'group': labels, 'x': umap_points[:, 0], 'y': umap_points[:, 1]}).to_csv(vis_dir / 'umap_points.csv', index=False, encoding='utf-8-sig')
    except Exception as exc:
        (vis_dir / 'umap_not_generated.txt').write_text(f'UMAP 未生成：{exc}', encoding='utf-8')

    print(f'增强可视化与质控结果已保存至：{vis_dir}')

# ===== 7. 训练与验证 =====
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, min_delta, fold, result_dir):
    best_model = None
    best_f1 = -1.0
    best_acc = 0.0
    early_stop_counter = 0
    best_val_preds = []
    best_val_labels = []
    best_val_probs = []
    fold_weight_path = os.path.join(result_dir, f'fold_{fold + 1}_best_weights.pth')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_probs = []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                probs = F.softmax(outputs, dim=1)
                val_loss += loss.item() * X.size(0)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        metrics = compute_all_metrics(val_labels, val_preds, val_probs, window_sec=CONFIG['window_sec'])
        current_f1 = metrics['spike_f1']
        current_acc = metrics['accuracy']

        if current_f1 > best_f1 + min_delta:
            best_f1 = current_f1
            best_acc = current_acc
            best_model = copy.deepcopy(model)
            best_val_preds = val_preds
            best_val_labels = val_labels
            best_val_probs = val_probs
            torch.save(best_model.state_dict(), fold_weight_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'[第 {fold + 1} 折] 早停：epoch {epoch + 1}，patience={patience}，min_delta={min_delta}')
                break

        print(f'[第 {fold + 1} 折] Epoch {epoch + 1:02d}/{epochs} | 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | Spike-F1: {current_f1:.4f} | Macro-F1: {metrics["macro_f1"]:.4f} | Specificity: {metrics["specificity"]:.4f} | PR-AUC: {metrics["pr_auc"]:.4f}')
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return best_model, fold_weight_path, best_f1, best_acc, best_val_preds, best_val_labels, best_val_probs


def run_cross_validation(config):
    set_seed(config['seed'])
    result_dir = Path(config['result_dir'])
    predictions_dir = result_dir / 'predictions'
    result_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    reproduction_note = f"""# EEGPT pretrained fine-tuning SOTA baseline notes

Experiment ID: {config['experiment_id']}
Reference paper: {config['reference_paper']}
Reference repository: {config['reference_repo']}
Checkpoint: {config['pretrained_checkpoint_path']}
Pretraining: {config['pretraining']}

Fair downstream comparison constraints:
- Same patient-level GroupKFold split as the main experiment.
- Same cached 0.2 s / 500 Hz z-scored raw windows and metadata.
- Same positive-sample augmentation policy: {config['augmentation_strategy']} with repeats={config['augmentation_repeats']}.
- Same downstream train/validation windows.
- Same Focal Loss: alpha={config['focal_alpha']}, gamma={config['focal_gamma']}.
- Same Adam training budget: lr={config['learning_rate']}, weight_decay={config['weight_decay']}, epochs={config['epochs']}, patience={config['patience']}.

Adaptation details:
- EEGPT is used as a pretrained foundation model, initialized from the local official checkpoint.
- This variant attaches the downstream classification head directly to mean-pooled pretrained target encoder representations after mean pooling over temporal patches and summary tokens; the reconstruction module used in self-supervised pretraining is not used in the downstream forward pass.
- The checkpoint was pretrained for 58-channel, 256 Hz, 4 s EEG with patch size 64.
- This study uses 16-channel, 0.2 s windows at 500 Hz. Each 100-sample window is linearly interpolated to {config['eegpt_img_time_len']} samples inside EEGPTEncoderHeadClassifier for compatibility with the pretrained patch encoder.
- With patch_size={config['eegpt_patch_size']} and patch_stride={config['eegpt_patch_stride']}, the adapted 256-sample input yields floor((256 - 64) / 32) + 1 = 7 temporal patches.
- This interpolation is only an input-length adaptation for the pretrained encoder; it is not interpreted as physiological resampling and does not extend the actual EEG context to the original 4 s pretraining window.
- EEGPT expects raw EEG input, so this baseline uses the standardized raw windows from the shared cache rather than the handcrafted raw/diff/SNE four-plane tensor used by the custom CNN baselines.
- T3/T4/T5/T6 are mapped to EEGPT's modern aliases T7/T8/P7/P8 for channel embeddings; signal order is unchanged.
- The target encoder and classification head are fine-tuned under the shared downstream training loop; all reconstructor parameters are frozen and excluded from the downstream computation path.
- No test-fold hyperparameter search is performed.

BibTeX placeholder:
Please cite the EEGPT NeurIPS 2024 paper according to the official repository/paper metadata.
"""
    (result_dir / 'EEGPT_pretrained_finetune_notes.md').write_text(reproduction_note, encoding='utf-8')

    X_all, y_all, metadata_df = load_window_cache(config)
    # 绘图已拆到 scripts/plot_augmentation_qc.py；训练阶段不再自动生成可视化。
    # visualize_augmentation_quality_from_cache(X_all, y_all, config, result_dir)

    gkf = GroupKFold(n_splits=config['n_splits'])
    groups = metadata_df['patient_id'].astype(str).to_numpy()
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups=groups)):
        print('\n' + '=' * 60)
        print(f'第 {fold + 1}/{config["n_splits"]} 折')

        train_X = X_all[train_idx]
        train_y = y_all[train_idx]
        val_X = X_all[val_idx]
        val_y = y_all[val_idx]
        val_meta = metadata_df.iloc[val_idx].drop(columns=['y'], errors='ignore').to_dict('records')

        if config['augment_positive'] and config['augmentation_strategy'] != 'none':
            pos_idx = np.where(train_y == 1)[0]
            augmented_samples = []
            for idx in pos_idx:
                for _ in range(config['augmentation_repeats']):
                    augmented_samples.append(augment(train_X[idx], config))
            if len(augmented_samples) > 0:
                augmented_X = np.array(augmented_samples)
                train_X = np.concatenate([train_X, augmented_X], axis=0)
                train_y = np.concatenate([train_y, np.ones(len(augmented_X), dtype=train_y.dtype)], axis=0)
                print(f'增强策略：{config["augmentation_strategy"]}；新增正样本：{len(augmented_X)}')
        else:
            print('增强策略：不增强；未新增正样本')

        print(f'训练集：{train_X.shape[0]} 个样本，正类={np.sum(train_y == 1)}，负类={np.sum(train_y == 0)}')
        print(f'验证集：{val_X.shape[0]} 个样本，正类={np.sum(val_y == 1)}，负类={np.sum(val_y == 0)}')

        train_loader = DataLoader(EEGDataset(train_X, train_y), batch_size=config['batch_size'], shuffle=True, drop_last=False)
        val_loader = DataLoader(EEGDataset(val_X, val_y), batch_size=config['batch_size'], shuffle=False, drop_last=False)

        model = FusionModel(config).to(device)
        print(f'EEGPT 特征形状：{model.feature_map_shape}；嵌入维度：{model.feature_dim}')
        with open(result_dir / f'fold_{fold + 1}_pretrained_load_report.json', 'w', encoding='utf-8') as f:
            json.dump(model.pretrained_load_report, f, ensure_ascii=False, indent=2, default=str)
        print(f'可训练参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

        criterion = FocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        best_model, fold_weight_path, best_f1, best_acc, val_preds, val_labels, val_probs = train_model(
            model, train_loader, val_loader, criterion, optimizer, config['epochs'], config['patience'], config['early_stop_min_delta'], fold, str(result_dir)
        )

        metrics = compute_all_metrics(val_labels, val_preds, val_probs, window_sec=config['window_sec'])
        metrics['fold'] = fold + 1
        metrics['weight_path'] = fold_weight_path
        fold_results.append(metrics)

        pred_df = pd.DataFrame(val_meta)
        pred_df.insert(0, 'fold', fold + 1)
        pred_df['y_true'] = val_labels
        pred_df['y_pred'] = val_preds
        pred_df['prob_spike'] = val_probs
        pred_df['prob_non_spike'] = 1 - pred_df['prob_spike']
        pred_df.to_csv(predictions_dir / f'fold_{fold + 1}_predictions.csv', index=False, encoding='utf-8-sig')

        print(pd.DataFrame(classification_report(val_labels, val_preds, output_dict=True, zero_division=0)).transpose().round(3))
        print(f"本折关键指标：Macro-F1={metrics['macro_f1']:.4f}，Spike-Recall={metrics['spike_recall_sensitivity']:.4f}，Specificity={metrics['specificity']:.4f}，FPR={metrics['fpr']:.4f}，PR-AUC={metrics['pr_auc']:.4f}")
        print(f'权重路径：{fold_weight_path}')

    fold_summary_df = pd.DataFrame(fold_results)
    ordered_cols = ['fold', 'TP', 'FP', 'TN', 'FN', 'support_non_spike', 'support_spike', 'accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall', 'weighted_f1', 'weighted_specificity_WS', 'spike_precision', 'spike_recall_sensitivity', 'spike_f1', 'specificity', 'fpr', 'fnr', 'fdr', 'npv', 'pr_auc', 'roc_auc', 'fp_per_min', 'fp_per_hour', 'weight_path']
    fold_summary_df = fold_summary_df[[c for c in ordered_cols if c in fold_summary_df.columns]]
    aggregate_df, paper_main_df, reviewer_df = build_summary_tables(fold_summary_df, config)
    save_all_tables(result_dir, config, fold_summary_df, aggregate_df, paper_main_df, reviewer_df)

    print('\n' + '=' * 60)
    print('交叉验证汇总')
    print(paper_main_df)
    print(f'全部表格已保存至：{result_dir}')
    print('推荐正文表：paper_main_table.csv')
    print('推荐补充逐折表：supplement_per_fold_table.csv')
    print('审稿人完整指标表：reviewer_metrics_table.csv')
    print('逐窗口预测明细：predictions/fold_x_predictions.csv')
    print('EEGPT 预训练微调说明：EEGPT_pretrained_finetune_notes.md')
    print('增强质控绘图请单独运行：scripts/plot_augmentation_qc.py')
    return fold_summary_df, aggregate_df, paper_main_df, reviewer_df

# ===== 8. 开始运行当前实验 =====
if __name__ == '__main__':
    run_cross_validation(CONFIG)
