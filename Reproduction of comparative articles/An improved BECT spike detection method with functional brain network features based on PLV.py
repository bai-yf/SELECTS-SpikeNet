import os
import numpy as np
import pandas as pd
import mne
from scipy.signal import hilbert
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  # Suppress mne warnings

# ==============================================================================
# 1. Configuration Parameters
# ==============================================================================
DATA_CONFIG = {
    "meta_path": "path/to/Train model/metadata/spick_time_label.xlsx",  # Path to metadata file
    "data_dir": "path/to/Train model/data",  # EEG data directory
    "sfreq": 500,  # Sampling frequency (Hz)
    "n_total_channels": 16,  # 16-channel central reference montage
    "channel_names": ['FP1','FP2','F3','F4','C3','C4','P3','P4',  # 16 channel names
                      'O1','O2','F7','F8','T3','T4','T5','T6']
}

METHOD_CONFIG = {
    # Chatrian template parameters
    "chatrian_window_ms": 200,    # Chatrian template window length
    "chatrian_target_amp_uv": 200, # Chatrian template target amplitude
    
    # Cross-correlation and morphology filtering parameters
    "corr_threshold": 0.6,        # Cross-correlation threshold 60%
    "morph_threshold": 0.3,       # Morphology feature threshold 30%
    
    # Peak-to-peak phenomenon screening parameters
    "peak_sync_threshold_ms": 20, # Peak synchronization time threshold ≤20ms
    
    # PLV functional brain network parameters
    "plv_window_ms": 100,         # PLV sliding window 100ms
    "plv_conn_threshold": 0.5,    # PLV > 0.5 considered as effective connection
    "core_network_nodes": ["F3","C3","P3","F4","C4","P4","T3","T4"],  # Core network nodes
    
    # ANN classifier parameters
    "input_dim": 35,              # 30 time-domain features + 5 network features
    "hidden1_dim": 64,            # First hidden layer 64 nodes
    "hidden2_dim": 32,            # Second hidden layer 32 nodes
    "batch_size": 64,
    "epochs": 100,                # 100 training epochs
    "lr": 1e-4,
    "patience": 5,                # Early stopping patience
    "dropout_rate": 0.5,          # Dropout to prevent overfitting
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Channel mapping (add AV_TO_BP_MAP for FPS validation to match monopolar-bipolar montage)
CHANNEL_MAP = {
    "core_av": ["C3", "P3"],  # Core AV channels
    "av_bp": {                 # AV channels → associated BP montage (original logic retained for initial peak-to-peak detection)
        "C3": ["F3-C3", "C3-P3"],
        "P3": ["C3-P3", "P3-O1"]
    },
    "AV_TO_BP_MAP": {          # Added: monopolar channel → bipolar montage mapping (for FPS validation)
        "F3": ["FP1-F3", "F3-C3"],
        "C3": ["F3-C3", "C3-P3"],
        "P3": ["C3-P3", "P3-O1"],
        "F4": ["FP2-F4", "F4-C4"],
        "C4": ["F4-C4", "C4-P4"],
        "P4": ["C4-P4", "P4-O2"],
        "F7": ["FP1-F7", "F7-T3"],
        "T3": ["F7-T3", "T3-T5"],
        "T5": ["T3-T5", "T5-O1"],
        "F8": ["FP2-F8", "F8-T4"],
        "T4": ["F8-T4", "T4-T6"],
        "T6": ["T4-T6", "T6-O2"],
        "C3+P3": ["F3-C3", "P3-O1"],
        "C4+P4": ["F4-C4", "P4-O2"],
        "F3+C3+P3": ["FP1-F3", "P3-O1"],
        "F4+C4+P4": ["FP2-F4", "P4-O2"]
    }
}

# Convert time parameters to sampling points
METHOD_CONFIG["chatrian_window"] = int(METHOD_CONFIG["chatrian_window_ms"] * DATA_CONFIG["sfreq"] / 1000)
METHOD_CONFIG["peak_sync_threshold"] = int(METHOD_CONFIG["peak_sync_threshold_ms"] * DATA_CONFIG["sfreq"] / 1000)
METHOD_CONFIG["plv_window"] = int(METHOD_CONFIG["plv_window_ms"] * DATA_CONFIG["sfreq"] / 1000)

# ==============================================================================
# 2. Data Loading
# ==============================================================================
def load_and_preprocess_data(meta_path, data_dir, sfreq, channel_names):
    """Load and preprocess patient EEG data"""
    meta_df = pd.read_excel(meta_path, header=None)
    patient_list = []
    
    for idx, row in meta_df.iterrows():
        # Extract patient ID and EEG file path
        pid_full = row[0].strip()
        pid = pid_full[:11]
        eeg_path = os.path.join(data_dir, f"{pid_full}.set")
        
        # Skip non-existent files
        if not os.path.exists(eeg_path):
            print(f"Warning: EEG file for patient {pid} does not exist, skipping")
            continue
        
        # Read EEG data
        raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
        
        # Verify channel order
        if raw.ch_names != channel_names:
            print(f"Warning: Patient {pid} channel order mismatch (actual: {raw.ch_names[:5]}...), skipping")
            continue
        raw_data = raw.get_data()[:,:600000]  # (n_channels, n_times), unit: V
        # Normalization
        raw_data = (raw_data - np.mean(raw_data, axis=1, keepdims=True)) / np.std(raw_data, axis=1, keepdims=True)
        n_times = raw_data.shape[1]
        
        # Extract true spike time points
        true_spike_times_sec = row[1:].dropna().astype(float).tolist()
        if not true_spike_times_sec:
            print(f"Warning: Patient {pid} has no true spike annotations, skipping")
            continue
        
        # Store patient data (added train_samples for FPS validation to split training set background)
        train_samples = int(n_times * 0.5)  # Use first 50% data as training set for background noise calculation
        patient_list.append({
            "pid": pid,
            "raw_data": raw_data,
            "true_spike_times_sec": true_spike_times_sec,
            "n_times": n_times,
            "train_samples": train_samples,  # Training sample count (for FPS background noise calculation)
            "channel_idx": {ch: channel_names.index(ch) for ch in channel_names}
        })
    
    print(f"\nSuccessfully loaded {len(patient_list)} patient data files (data dimension: {raw_data.shape})\n")
    return patient_list

# ==============================================================================
# 3. Core Utility Functions (added calculate_bp_peak_to_peak, modified FPS validation logic)
# ==============================================================================
def create_chatrian_template(fs, window_ms, raw_data, target_amplitude_uv=200):
    """Create Chatrian template"""
    window_samples = int(window_ms * fs / 1000)
    t = np.linspace(-window_ms/2, window_ms/2, window_samples)
    
    target_amplitude_v = np.max(raw_data)
    spike_peak = -target_amplitude_v * np.exp(-(t/30)** 2)
    slow_wave_forward = target_amplitude_v * 0.25 * np.exp(-((t + 20)/15)**2)  
    slow_wave_backward = target_amplitude_v * 0.25 * np.exp(-((t - 20)/15)** 2)  

    spike_template = -(spike_peak + slow_wave_forward + slow_wave_backward)
    spike_template -= np.mean(spike_template)
    std = np.std(spike_template)
    if std != 0:
        spike_template /= std
    return spike_template, t

def correlate(x, y, mode='valid'):
    """Custom cross-correlation function"""
    if mode != 'valid':
        raise ValueError("Currently only supports mode='valid'")
    
    len_x = len(x)
    len_y = len(y)
    if len_x < len_y:
        raise ValueError(f"Signal length ({len_x}) must ≥ template length ({len_y})")
    
    corr_len = len_x - len_y + 1
    corr_result = np.zeros(corr_len, dtype=np.float64)
    
    # Sliding window cross-correlation calculation
    for i in range(corr_len):
        x_window = x[i:i + len_y]
        corr_value = np.sum(x_window/(np.max(x_window)-np.min(x_window)) * y)
        corr_result[i] = corr_value
    
    return corr_result

def calc_morph_features(seg):
    """Calculate morphology features"""
    peak_idx = np.argmax(np.abs(seg))  # Peak position
    n = len(seg)
    max_amp = np.max(np.abs(seg)) + 1e-6  # Avoid division by zero
    
    # Rise/fall slope
    rise_slope = (seg[peak_idx] - seg[0]) / (peak_idx + 1e-6)
    fall_slope = (seg[-1] - seg[peak_idx]) / (n - peak_idx - 1 + 1e-6)
    slope_norm = (np.abs(rise_slope) + np.abs(fall_slope)) / (2 * max_amp)
    
    # Curvature
    curvature = np.mean(np.abs(np.diff(seg, n=2))) / max_amp
    
    return (slope_norm + curvature) / 2  # Mean morphology feature

def generate_bipolar_signals(raw_data, channel_idx, av_bp_map):
    """Generate bipolar montage signals (original logic retained for initial peak-to-peak detection)"""
    bp_signals = {}
    av_bp_valid = {av: [] for av in av_bp_map.keys()}
    
    for av, bp_list in av_bp_map.items():
        if av not in channel_idx:
            continue
        for bp in bp_list:
            ch_prev, ch_curr = bp.split("-")
            if ch_prev not in channel_idx or ch_curr not in channel_idx:
                continue
            # Calculate BP signal: posterior channel potential - anterior channel potential (original logic)
            bp_signals[bp] = raw_data[channel_idx[ch_curr]] - raw_data[channel_idx[ch_prev]]
            av_bp_valid[av].append(bp)
    
    av_bp_valid = {av: bps for av, bps in av_bp_valid.items() if bps}
    return bp_signals, av_bp_valid

def check_peak_to_peak(av_seg, bp_segs, sync_threshold):
    """Check peak-to-peak phenomenon (original logic retained for initial screening)"""
    av_peak_idx = np.argmax(np.abs(av_seg))
    sync_bps = []
    for bp_name, bp_seg in bp_segs.items():
        bp_peak_idx = np.argmax(np.abs(bp_seg))
        if abs(bp_peak_idx - av_peak_idx) <= sync_threshold:
            sync_bps.append(bp_name)
    return len(sync_bps) >= 1, sync_bps

# ---------------------- Added: Bipolar montage peak-to-peak calculation function ----------------------
def calculate_bp_peak_to_peak(bp_signal, peak_center_idx, fs, window_ms=100):
    """Calculate peak-to-peak amplitude in bipolar montage signal within peak center window (FPS validation core function)"""
    window_samples = int(window_ms * fs / 1000)
    # Calculate window start/end index (centered on peak)
    start_idx = max(0, peak_center_idx - window_samples // 2)
    end_idx = min(bp_signal.shape[-1], peak_center_idx + window_samples // 2)
    # Extract window signal
    window_signal = bp_signal[start_idx:end_idx]
    # Calculate peak-to-peak amplitude (maximum - minimum)
    return np.max(window_signal) - np.min(window_signal)

# ---------------------- Added: FPS (Bipolar Montage Validation) Function ----------------------
def fps_validate_candidates(raw_data, channel_idx, merged_candidates, fs, spike_window_ms, train_samples):
    """
    FPS (Bipolar Montage Validation): Remove invalid candidate spikes
    Logic: 1. Generate all bipolar montage signals 2. Validate peak-to-peak amplitude of bipolar montages for candidate spikes
    """
    # 1. Generate all bipolar montage signals (according to specified montage combinations)
    bp_channels = {}
    bp_combinations = [("F3","C3"), ("C3","P3"), ("F4","C4"), ("C4","P4"), 
                       ("F7","T3"), ("T3","T5"), ("F8","T4"), ("T4","T6"),
                       ("FP1","F3"), ("P3","O1"), ("FP2","F4"), ("P4","O2")]
    
    for av_ch1, av_ch2 in bp_combinations:
        bp_name = f"{av_ch1}-{av_ch2}"
        # Bipolar montage calculation: posterior channel (av_ch2) - anterior channel (av_ch1) (according to specified logic)
        bp_data = raw_data[channel_idx[av_ch2]] - raw_data[channel_idx[av_ch1]]
        bp_channels[bp_name] = bp_data
    
    # 2. Validate each candidate spike
    valid_candidates = []
    AV_TO_BP_MAP = CHANNEL_MAP["AV_TO_BP_MAP"]  # Monopolar-bipolar montage mapping
    
    for cand in merged_candidates:
        t_start, t_end = cand
        # Candidate spike center index (for locating bipolar montage window)
        center_idx = (t_start + t_end) // 2
        # Candidate spike belongs to monopolar channel (default to core AV channel, simplified to C3/P3 here, adjust according to actual scenario)
        # In actual scenarios, need to first determine candidate's channel, simplified here to prioritize matching C3, otherwise P3
        ch_name = "C3" if "C3" in channel_idx else "P3"
        # Get bipolar montage combinations corresponding to this monopolar channel
        bp_names = AV_TO_BP_MAP.get(ch_name, [])
        
        # Need 2 bipolar montage combinations to perform validation
        if len(bp_names) != 2:
            continue
        
        # Validate peak-to-peak amplitude for each bipolar montage
        bp_valid = []
        for bp_name in bp_names:
            if bp_name not in bp_channels:
                continue
            # Get bipolar montage data
            bp_data = bp_channels[bp_name]
            # Calculate peak-to-peak amplitude (window length = spike window length)
            p2p = calculate_bp_peak_to_peak(
                bp_signal=bp_data,
                peak_center_idx=center_idx,
                fs=fs,
                window_ms=spike_window_ms
            )
            # Calculate background noise standard deviation (use last 50% of training data to avoid true spike interference)
            bg_std = np.std(bp_data[train_samples:]) if train_samples > 0 else np.std(bp_data)
            # Validation condition: peak-to-peak amplitude > 1.5× background noise (according to specified threshold)
            bp_valid.append(p2p > 1.5 * bg_std)
        
        # Only consider as valid candidate if both bipolar montages pass validation
        if all(bp_valid):
            valid_candidates.append(cand)
    
    return valid_candidates, bp_channels

def plot_waveform_template_alignment(template, waveform, template_name, channel_name, 
                                     fs, window_ms=200, patient_id="Unknown", save_path=None):
    """Plot waveform vs template alignment comparison"""
    # Ensure template and waveform have same length
    window_samples = max(len(template), len(waveform))
    if len(template) < window_samples:
        pad_left = (window_samples - len(template)) // 2
        pad_right = window_samples - len(template) - pad_left
        template = np.pad(template, (pad_left, pad_right), mode='constant')
    if len(waveform) < window_samples:
        pad_left = (window_samples - len(waveform)) // 2
        pad_right = window_samples - len(waveform) - pad_left
        waveform = np.pad(waveform, (pad_left, pad_right), mode='constant')
    
    # Calculate relative time axis (seconds)
    t_relative = np.linspace(-window_ms/2, window_ms/2, window_samples) / 1000
    
    # Calculate correlation
    corr = correlate(waveform, template, mode='valid')[0]
    corr_norm = corr / (np.std(waveform) * np.std(template) * len(template) + 1e-8)
    corr_norm /= np.max(np.abs(corr_norm)) if np.max(np.abs(corr_norm)) != 0 else 1
    print(f"Channel {channel_name}: Waveform vs template {template_name} normalized correlation = {corr_norm:.3f}")
    
    # Find peak positions
    wave_peak_idx = np.argmax(np.abs(waveform))
    temp_peak_idx = np.argmax(np.abs(template))
    
    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(
        f'Patient {patient_id} - Waveform vs Template Alignment (Channel: {channel_name})\n'
        f'Template: {template_name} | Window: {window_ms}ms | Correlation: {corr_norm:.3f}',
        fontsize=14, y=1.02
    )
    
    # Plot EEG waveform
    ax_wave = axes[0]
    ax_wave.plot(t_relative, waveform, color='blue', linewidth=1.5, label='EEG Waveform')
    ax_wave.scatter(t_relative[wave_peak_idx], waveform[wave_peak_idx], 
                   color='red', s=60, zorder=5, label='Waveform Peak')
    ax_wave.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Window Center')
    ax_wave.set_title('EEG Waveform Segment', fontsize=12)
    ax_wave.set_ylabel('Amplitude', fontsize=10)
    ax_wave.grid(linestyle=':', alpha=0.3)
    ax_wave.legend(fontsize=9)
    
    # Plot template
    ax_temp = axes[1]
    ax_temp.plot(t_relative, template, color='orange', linewidth=1.5, label='Template')
    ax_temp.scatter(t_relative[temp_peak_idx], template[temp_peak_idx], 
                   color='green', s=60, zorder=5, label='Template Peak')
    ax_temp.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax_temp.set_title(f'Template {template_name}', fontsize=12)
    ax_temp.set_xlabel('Relative Time (s)', fontsize=10)
    ax_temp.set_ylabel('Amplitude', fontsize=10)
    ax_temp.grid(linestyle=':', alpha=0.3)
    ax_temp.legend(fontsize=9)
    
    # Set x-axis range
    axes[-1].set_xlim(-window_ms/2000 - 0.01, window_ms/2000 + 0.01)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# ==============================================================================
# 4. Candidate Spike Detection (adjusted to output only template-matched candidates, FPS validation handled in subsequent steps)
# ==============================================================================
def detect_candidate_spikes(patient_data, method_config, channel_map):
    """Complete candidate spike detection pipeline (refactored: find positions first → then extract segments, output only template-matched candidates)"""
    raw_data = patient_data["raw_data"]
    channel_idx = patient_data["channel_idx"]
    n_times = patient_data["n_times"]
    pid = patient_data["pid"]
    fs = DATA_CONFIG["sfreq"]
    spike_window_ms = method_config["chatrian_window_ms"]  # Candidate segment total length (e.g., 200ms)
    half_window = int(spike_window_ms * fs / 1000) // 2  # Half window length (for center alignment)
    template_len = method_config["chatrian_window"]  # Chatrian template length (same as half window)

    # Step 1: Generate Chatrian template
    core_av = channel_map["core_av"][0]
    if core_av not in channel_idx:
        print(f"Patient {pid}: Core channel {core_av} does not exist, cannot generate Chatrian template")
        return [], {}, {}
    core_ch_idx = channel_idx[core_av]
    core_ch_data = raw_data[core_ch_idx]
    chatrian_template, _ = create_chatrian_template(
        fs=fs,
        window_ms=spike_window_ms,
        raw_data=core_ch_data,
        target_amplitude_uv=method_config["chatrian_target_amp_uv"]
    )
    print(f"Patient {pid}: Generated Chatrian template based on {core_av} channel (length: {len(chatrian_template)} samples)")

    # Step 2: Generate BP signals (for subsequent peak-to-peak validation)
    av_bp_map = channel_map["av_bp"]
    bp_signals_init, av_bp_valid = generate_bipolar_signals(raw_data, channel_idx, av_bp_map)
    if not av_bp_valid:
        print(f"Patient {pid}: No valid AV-BP mapping, cannot detect candidate spikes")
        return [], bp_signals_init, av_bp_valid

    # ---------------------- Key modification: Find matching positions first, then extract segments ----------------------
    # Step 3: Full signal cross-correlation to find all high similarity positions (not limited to training set to avoid missed detections)
    av_data_full = core_ch_data  # Use full core channel data for matching (not just training set)
    # Calculate cross-correlation of full data (step size = 1, ensure no positions are missed)
    corr_raw_full = correlate(av_data_full, chatrian_template, mode='valid')
    # Normalize cross-correlation results
    corr_norm_full = corr_raw_full / (np.std(av_data_full) * np.std(chatrian_template) * template_len + 1e-8)
    corr_norm_full /= np.max(np.abs(corr_norm_full)) if np.max(np.abs(corr_norm_full)) != 0 else 1

    # Step 4: Filter positions exceeding cross-correlation threshold (these are potential spike centers)
    high_corr_positions = np.where(corr_norm_full >= method_config["corr_threshold"])[0]
    if len(high_corr_positions) == 0:
        print(f"Patient {pid}: No positions found exceeding cross-correlation threshold, no candidate spikes")
        return [], bp_signals_init, av_bp_valid
    print(f"Patient {pid}: Number of positions exceeding cross-correlation threshold: {len(high_corr_positions)}")

    # Step 5: Remove adjacent duplicates (avoid same spike detected multiple times)
    # Adjacent positions spaced < half window length (i.e., 100ms) considered same spike, take middle position
    unique_positions = []
    if len(high_corr_positions) > 0:
        current_pos = high_corr_positions[0]
        for pos in high_corr_positions[1:]:
            if pos - current_pos > half_window:  # Spacing exceeds half window, considered new spike
                unique_positions.append(current_pos)
                current_pos = pos
        unique_positions.append(current_pos)  # Add last position
    print(f"Patient {pid}: Number of potential spike centers after deduplication: {len(unique_positions)}")

    # Step 6: Extract candidate segments centered on each position (ensure segment contains complete spike)
    initial_candidates = []
    for center_pos in unique_positions:
        # Calculate segment start/end indices (center ± half window, ensure not out of bounds)
        t_start = max(0, center_pos - half_window)
        t_end = min(n_times, center_pos + half_window)
        # Extract AV channel segment
        av_seg = av_data_full[t_start:t_end]
        # Morphology feature filtering (remove segments with mismatched morphology)
        # morph_score = calc_morph_features(av_seg)
        # if morph_score < method_config["morph_threshold"]:
        #     continue
        # Peak-to-peak validation (using BP signals)
        av_channel = core_av
        bp_list = av_bp_valid[av_channel]
        bp_segs = {bp: bp_signals_init[bp][t_start:t_end] for bp in bp_list}
        has_peak2peak, sync_bps = check_peak_to_peak(av_seg, bp_segs, method_config["peak_sync_threshold"])
        if not has_peak2peak:
            continue
        # Add qualifying segments to candidates
        initial_candidates.append((t_start, t_end, av_channel, sync_bps))
    print(f"Patient {pid}: Chatrian template initial screening candidate count (center extraction + multi-condition filtering): {len(initial_candidates)}")

    # Subsequent K-Means clustering, FPS validation logic unchanged (only input candidate segments updated)
    if not initial_candidates:
        return [], bp_signals_init, av_bp_valid

    # Step 7: K-Means clustering optimization (modified part)
    cluster_features = []
    valid_peaks = []
    # Define standard length (must be fixed, e.g., 2*half_window)
    standard_len = 2 * half_window  # Half window * 2 = full window length
    
    for t_start, t_end, av_ch, sync_bps in initial_candidates:
        av_idx = channel_idx[av_ch]
        seg = raw_data[av_idx, t_start:t_end].copy()
        
        # 1. Standardize segment length (core fix)
        current_len = len(seg)
        if current_len < standard_len:
            # Length insufficient: pad zeros (at end, or center-aligned)
            pad_length = standard_len - current_len
            seg = np.pad(seg, (0, pad_length), mode='constant')  # Pad zeros at end
            # For center alignment: seg = np.pad(seg, (pad_length//2, pad_length - pad_length//2), mode='constant')
        elif current_len > standard_len:
            # Length too long: truncate (keep middle part)
            seg = seg[(current_len - standard_len) // 2 : (current_len + standard_len) // 2]
        
        # 2. Normalization, alignment (original logic)
        seg_norm = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
        peak_idx = np.argmax(np.abs(seg_norm))
        if seg_norm[peak_idx] < 0:
            seg_norm = -seg_norm  # Force peak upward
        shift = half_window - peak_idx
        seg_aligned = np.roll(seg_norm, shift)
        
        # 3. Ensure aligned length still standard length (secondary verification)
        if len(seg_aligned) != standard_len:
            # Edge case: truncate or pad again
            seg_aligned = seg_aligned[:standard_len] if len(seg_aligned) > standard_len else np.pad(seg_aligned, (0, standard_len - len(seg_aligned)), mode='constant')
        
        cluster_features.append(seg_aligned.flatten())
        valid_peaks.append((t_start, t_end, av_ch, seg_aligned))

    # Now converting to array won't cause errors
    cluster_features = np.array(cluster_features)
    max_clusters = min(3, len(cluster_features))
    best_n_clusters = 1
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        labels = kmeans.fit_predict(cluster_features)
        cluster_sizes = np.bincount(labels)
        if any(size < 0.05 * len(cluster_features) for size in cluster_sizes):
            best_n_clusters = n_clusters - 1 if n_clusters > 1 else 1
            break
        best_n_clusters = n_clusters

    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=3)
    kmeans.fit(cluster_features)
    print(f"Patient {pid}: K-Means cluster count: {best_n_clusters} (generated {best_n_clusters} specific templates)")

    # Step 8: Extract only upward-peaking cluster templates, generate candidates
    final_candidates = []
    for cluster_id in range(best_n_clusters):
        cluster_mask = (kmeans.labels_ == cluster_id)
        cluster_peaks = [valid_peaks[i] for i in np.where(cluster_mask)[0]]
        if not cluster_peaks:
            continue
        # Cluster center (already ensured peak upward)
        cluster_center = kmeans.cluster_centers_[cluster_id].reshape(-1)
        # Re-match full data using cluster center
        corr_raw_temp = correlate(av_data_full, cluster_center[:template_len], mode='valid')
        corr_norm_temp = corr_raw_temp / (np.std(av_data_full) * np.std(cluster_center) * template_len + 1e-8)
        corr_norm_temp /= np.max(np.abs(corr_norm_temp)) if np.max(np.abs(corr_norm_temp)) != 0 else 1
        temp_high_pos = np.where(corr_norm_temp >= method_config["corr_threshold"])[0]
        # Deduplicate then extract segments
        temp_unique_pos = []
        if len(temp_high_pos) > 0:
            curr_p = temp_high_pos[0]
            for p in temp_high_pos[1:]:
                if p - curr_p > half_window:
                    temp_unique_pos.append(p)
                    curr_p = p
            temp_unique_pos.append(curr_p)
        # Add to final candidates
        for p in temp_unique_pos:
            t_s = max(0, p - half_window)
            t_e = min(n_times, p + half_window)
            final_candidates.append((t_s, t_e))

    print(f"Patient {pid}: Template matching merged candidate count: {len(final_candidates)}")

    # FPS validation will be handled in subsequent steps, only return template-matched candidates here
    return final_candidates, bp_signals_init, av_bp_valid

# ==============================================================================
# 5. Feature Extraction (add debugging logs)
# ==============================================================================
def extract_time_features(seg, seg_desc="Default segment"):
    """Extract 10 time-domain features from single channel segment (fix empty slice issue)"""
    n = len(seg)
    
    # Handle empty segments
    if n == 0:
        print(f"[ERROR] {seg_desc}: Empty array! Triggered zero-size error")
        return np.zeros(10)
    
    # Handle abnormal values (NaN/Inf)
    if np.isnan(seg).any() or np.isinf(seg).any():
        nan_count = np.sum(np.isnan(seg))
        inf_count = np.sum(np.isinf(seg))
        print(f"[ERROR] {seg_desc}: Contains {nan_count} NaN and {inf_count} Inf, replaced with valid values")
        seg_clean = np.nan_to_num(seg, nan=0.0, posinf=1.0, neginf=-1.0)
    else:
        seg_clean = seg.copy()
    
    try:
        peak_idx = np.argmax(np.abs(seg_clean))
        max_amp = np.max(np.abs(seg_clean)) + 1e-6  # Avoid division by zero
        
        # ---------------------- Fix 1: Handle empty slices (amp_left/amp_right) ----------------------
        # Left half-wave slice (0~peak_idx): if peak_idx=0, slice is [0:1] (non-empty)
        left_slice = seg_clean[:peak_idx+1]
        # Right half-wave slice (peak_idx+1~n-1): if peak_idx=n-1, slice is empty, default amplitude=0
        right_slice = seg_clean[peak_idx+1:]
        
        # Set default amplitude for empty slices (avoid np.max processing empty array)
        amp_left = np.max(np.abs(left_slice)) / max_amp if len(left_slice) > 0 else 0.0
        amp_right = np.max(np.abs(right_slice)) / max_amp if len(right_slice) > 0 else 0.0
        amp_peak = np.abs(seg_clean[peak_idx]) / max_amp
        
        # ---------------------- Fix 2: Handle boundary cases for slope calculation ----------------------
        # Rise slope: if peak_idx=0, no rising edge, slope=0
        slope_rise = np.abs((seg_clean[peak_idx] - seg_clean[0]) / (peak_idx + 1e-6)) / max_amp if peak_idx > 0 else 0.0
        # Fall slope: if peak_idx=n-1, no falling edge, slope=0
        slope_fall = np.abs((seg_clean[-1] - seg_clean[peak_idx]) / (n - peak_idx - 1 + 1e-6)) / max_amp if (n - peak_idx - 1) > 0 else 0.0
        # Peak adjacent slope: if peak_idx=0, no previous point, slope=0
        slope_peak = np.abs(seg_clean[peak_idx] - seg_clean[max(0, peak_idx-1)]) / max_amp if n > 1 else 0.0
        
        # Other feature calculations (normal)
        dura_left = (peak_idx + 1) / n
        dura_right = (n - peak_idx - 1) / n
        dura_full = n / n
        area_full = np.sum(np.abs(seg_clean)) / (max_amp * n)
        
        # Assemble features
        feat = np.array([dura_left, dura_right, dura_full,
                         amp_left, amp_right, amp_peak,
                         slope_rise, slope_fall, slope_peak,
                         area_full])
        return feat
    
    except Exception as e:
        print(f"[ERROR] {seg_desc}: Feature calculation failed! Error type={type(e).__name__}, error message={str(e)}")
        return np.zeros(10)
    
def get_time_domain_features(raw_data, bp_signals, candidates, av_bp_valid, channel_idx, patient_pid="Unknown"):
    """Extract 30-dimensional time-domain features (add debugging logs)"""
    time_features = []
    # Validate av_bp_valid effectiveness
    if not av_bp_valid:
        print(f"[ERROR] Patient {patient_pid}: av_bp_valid is empty, cannot extract time-domain features")
        return np.array([])
    
    av_channel = next(iter(av_bp_valid.keys()))  # Take first valid AV channel
    if av_channel not in channel_idx:
        print(f"[ERROR] Patient {patient_pid}: AV channel {av_channel} not in channel list")
        return np.array([])
    
    av_idx = channel_idx[av_channel]
    bp_channels = av_bp_valid[av_channel][:2]  # Take 2 BP channels
    
    # Extract features for each candidate
    for cand_idx, (t_start, t_end) in enumerate(candidates):
        # 1. Validate index effectiveness
        max_time = raw_data.shape[1]
        if t_start >= max_time or t_end > max_time or t_start >= t_end:
            continue
        
        # 2. Extract AV channel features
        av_seg = raw_data[av_idx, t_start:t_end]
        av_desc = f"AV channel {av_channel} candidate {cand_idx+1}"
        av_feat = extract_time_features(av_seg, seg_desc=av_desc)
        
        # 3. Extract BP channel features
        bp_feats = []
        valid_bp_count = 0
        for bp_idx, bp in enumerate(bp_channels):
            # Validate BP channel existence
            if bp not in bp_signals:
                continue
            # Extract BP segment
            bp_seg = bp_signals[bp][t_start:t_end]
            bp_desc = f"BP channel {bp} candidate {cand_idx+1}"
            
            # Handle empty BP segment
            if len(bp_seg) == 0:
                break
            
            # Calculate BP features
            bp_feat = extract_time_features(bp_seg, seg_desc=bp_desc)
            bp_feats.append(bp_feat)
            valid_bp_count += 1
        
        # 4. Ensure at least 2 valid BP channels
        if valid_bp_count < 2:
            continue
        
        # 5. Concatenate features
        time_feat = np.concatenate([av_feat, bp_feats[0], bp_feats[1]])
        time_features.append(time_feat)
    
    # Final validation
    time_features = np.array(time_features)
    return time_features

def calc_plv(x, y, x_desc="Signal X", y_desc="Signal Y"):
    """Calculate PLV (add debugging logs)"""
    if len(x) == 0 or len(y) == 0:
        return 0.0
    try:
        hilbert_x = hilbert(x)
        hilbert_y = hilbert(y)
        phase_x = np.angle(hilbert_x)
        phase_y = np.angle(hilbert_y)
        phase_diff = phase_x - phase_y
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        return plv
    except Exception as e:
        return 0.0

def build_plv_network(raw_data, t_start, t_end, plv_window, conn_threshold, core_nodes, channel_idx, patient_pid="Unknown"):
    """Build PLV functional brain network (add debugging logs)"""
    max_time = raw_data.shape[1]
    
    # 1. Validate segment effectiveness
    if t_start >= max_time or t_end > max_time or t_start >= t_end:
        return np.zeros(5)
    
    seg = raw_data[:, t_start:t_end]
    
    # 2. Validate core nodes
    valid_core_nodes = [ch for ch in core_nodes if ch in channel_idx]
    core_indices = [channel_idx[ch] for ch in valid_core_nodes]
    n_core = len(valid_core_nodes)
    
    if n_core < 2:
        return np.zeros(5)
    
    # 3. Calculate PLV matrix
    n_times = seg.shape[1]
    plv_matrix = np.zeros((n_core, n_core))
    win_count = (n_times - plv_window + 1) // plv_window
    
    for win_idx, win_start in enumerate(range(0, n_times - plv_window + 1, plv_window)):
        win_end = win_start + plv_window
        win_seg = seg[core_indices, win_start:win_end]
        
        for i in range(n_core):
            for j in range(i+1, n_core):
                plv = calc_plv(win_seg[i], win_seg[j])
                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv
    
    # Subsequent network feature calculation logic unchanged...
    adj_matrix = (plv_matrix > conn_threshold).astype(int)
    degree = np.sum(adj_matrix, axis=1)
    ad = np.mean(degree)
    
    acc = 0.0
    for i in range(n_core):
        neighbors = np.where(adj_matrix[i] == 1)[0]
        k = len(neighbors)
        if k < 2:
            continue
        actual_edges = 0
        for j in neighbors:
            for l in neighbors:
                if j < l and adj_matrix[j, l] == 1:
                    actual_edges += 1
        acc += 2 * actual_edges / (k * (k - 1))
    acc /= n_core
    
    inf = np.inf
    dist_matrix = np.where(adj_matrix == 0, inf, 1)
    for k in range(n_core):
        for i in range(n_core):
            for j in range(n_core):
                if dist_matrix[i, j] > dist_matrix[i, k] + dist_matrix[k, j]:
                    dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
    valid_dists = dist_matrix[dist_matrix != inf]
    cpl = np.mean(valid_dists) if len(valid_dists) > 0 else 0
    ge = np.mean(1 / valid_dists) if len(valid_dists) > 0 else 0
    
    module1_rel = list(range(n_core//2))
    module2_rel = list(range(n_core//2, n_core))
    total_edges = np.sum(adj_matrix) / 2
    md = 0.0
    if total_edges > 0:
        e11 = np.sum(adj_matrix[np.ix_(module1_rel, module1_rel)]) / 2
        e22 = np.sum(adj_matrix[np.ix_(module2_rel, module2_rel)]) / 2
        a1 = np.sum(adj_matrix[module1_rel]) / (2 * total_edges)
        a2 = np.sum(adj_matrix[module2_rel]) / (2 * total_edges)
        md = (e11/total_edges - a1**2) + (e22/total_edges - a2**2)
    
    network_feat = np.array([ad, acc, cpl, md, ge])
    return network_feat

def get_network_features(raw_data, candidates, method_config, channel_idx, patient_pid="Unknown"):
    """Extract 5-dimensional PLV network features (add debugging logs)"""
    network_features = []
    core_nodes = method_config["core_network_nodes"]
    plv_window = method_config["plv_window"]
    conn_threshold = method_config["plv_conn_threshold"]
    max_time = raw_data.shape[1]
    
    for cand_idx, (t_start, t_end) in enumerate(candidates):
        # Extend segment to PLV window length
        seg_len = t_end - t_start
        if seg_len < plv_window:
            need_extend = plv_window - seg_len
            new_t_end = t_end + need_extend
            new_t_start = t_start
            # Adjust to avoid out-of-bounds
            if new_t_end > max_time:
                new_t_start = max(0, t_end - plv_window)
                new_t_end = t_end
            t_start, t_end = new_t_start, new_t_end
        
        # Calculate network features
        network_feat = build_plv_network(
            raw_data, t_start, t_end, plv_window, conn_threshold, core_nodes, channel_idx, patient_pid=patient_pid
        )
        network_features.append(network_feat)
    
    network_features = np.array(network_features)
    return network_features

# ==============================================================================
# 6. Label Annotation and Sample Balancing (modified calls, pass patient ID)
# ==============================================================================
def create_feature_label(patient_list, method_config):
    """Fuse features and annotate labels (pass patient ID for debugging)"""
    X_all = []
    y_all = []
    sfreq = DATA_CONFIG["sfreq"]
    
    for patient in patient_list:
        pid = patient["pid"]
        if "valid_candidates" not in patient or not patient["valid_candidates"]:
            continue
        
        # Extract time-domain features (pass patient ID for debugging)
        time_feat = get_time_domain_features(
            patient["raw_data"], patient["bp_signals"], patient["valid_candidates"],
            patient["av_bp_valid"], patient["channel_idx"], patient_pid=pid
        )
        if len(time_feat) == 0:
            continue
        
        # Extract network features (pass patient ID for debugging)
        network_feat = get_network_features(
            patient["raw_data"], patient["valid_candidates"], method_config, patient["channel_idx"], patient_pid=pid
        )
        if len(network_feat) == 0:
            continue
        
        # Ensure consistent feature lengths
        min_len = min(len(time_feat), len(network_feat))
        if min_len == 0:
            continue
        fused_feat = np.hstack([time_feat[:min_len], network_feat[:min_len]])
        
        # Annotate labels
        true_spike_samples = [int(t * sfreq) for t in patient["true_spike_times_sec"]]
        labels = []
        for (t_start, t_end) in patient["valid_candidates"][:min_len]:
            is_true = any(t_start <= tsp <= t_end for tsp in true_spike_samples)
            labels.append(1 if is_true else 0)
        
        X_all.extend(fused_feat.tolist())
        y_all.extend(labels)
    
    # Sample balancing logic unchanged...
    if not X_all:
        raise ValueError("No valid feature samples, cannot train model")
    
    X = np.array(X_all)
    y = np.array(y_all)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(neg_idx) > len(pos_idx):
        neg_idx_downsampled = np.random.choice(neg_idx, len(pos_idx), replace=False)
        balanced_idx = np.concatenate([pos_idx, neg_idx_downsampled])
        X_balanced = X[balanced_idx]
        y_balanced = y[balanced_idx]
    else:
        X_balanced = X
        y_balanced = y
    
    print(f"\n[Feature sample after balancing] Total samples={len(X_balanced)} (positive={np.sum(y_balanced)}, negative={len(y_balanced)-np.sum(y_balanced)})")
    return X_balanced, y_balanced

# ==============================================================================
# 7. ANN Model Training and Evaluation (unchanged)
# ==============================================================================
class SpikeDataset(Dataset):
    """Spike feature dataset class"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # (n_samples, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SpikeANN(nn.Module):
    """Four-layer ANN model"""
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, dropout_rate):
        super(SpikeANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden2_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def calculate_specificity(cm):
    """Calculate specificity (SP)"""
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def train_ann_model(X_balanced, y_balanced, method_config):
    """Train ANN model"""
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    # Build DataLoader
    train_dataset = SpikeDataset(X_train, y_train)
    test_dataset = SpikeDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=method_config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=method_config["batch_size"], shuffle=False)
    
    # Initialize model
    model = SpikeANN(
        method_config["input_dim"], method_config["hidden1_dim"], 
        method_config["hidden2_dim"], method_config["dropout_rate"]
    ).to(method_config["device"])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=method_config["lr"])
    
    # Training parameters
    best_val_se = 0.0
    early_stop_count = 0
    train_losses = []
    
    # Training loop
    print(f"\nStarting ANN model training (device: {method_config['device']})")
    for epoch in range(method_config["epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(method_config["device"]), y_batch.to(method_config["device"])
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(method_config["device"])
                outputs = model(X_batch)
                y_pred.extend((outputs > 0.5).float().cpu().numpy())
                y_true.extend(y_batch.cpu().numpy())
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        # Calculate metrics
        cm = confusion_matrix(y_true, y_pred)
        val_ac = accuracy_score(y_true, y_pred)
        val_se = recall_score(y_true, y_pred)
        val_sp = calculate_specificity(cm)
        
        # Print logs
        print(f"Epoch {epoch+1:3d}/{method_config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val AC: {val_ac:.4f}, SE: {val_se:.4f}, SP: {val_sp:.4f}")
        
        # Early stopping logic
        if val_se > best_val_se:
            best_val_se = val_se
            early_stop_count = 0
            torch.save(model.state_dict(), "best_spike_ann.pth")
        else:
            early_stop_count += 1
            if early_stop_count >= method_config["patience"]:
                print(f"Early stopping: {method_config['patience']} consecutive epochs validation SE not improved, best SE: {best_val_se:.4f}")
                break
    
    # Final evaluation
    model.load_state_dict(torch.load("best_spike_ann.pth"))
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(method_config["device"])
        y_pred_final = (model(X_test_tensor) > 0.5).float().cpu().numpy()
    
    # Calculate final metrics
    final_cm = confusion_matrix(y_test, y_pred_final)
    final_ac = accuracy_score(y_test, y_pred_final)
    final_se = recall_score(y_test, y_pred_final)
    final_sp = calculate_specificity(final_cm)
    
    # Print results
    print("\n" + "="*60)
    print("Final Model Evaluation Results (based on test set)")
    print(f"Confusion Matrix:\n{final_cm}")
    print(f"Accuracy (AC): {final_ac:.4f}")
    print(f"Sensitivity (SE): {final_se:.4f}")
    print(f"Specificity (SP): {final_sp:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_final, digits=4))
    print("="*60)
    
    return model, (final_ac, final_se, final_sp)

def plot_template_correlation(template, channel_data, fs, window_ms=200, patient_id="Unknown", adaptive_thresh=0.6):
    """Plot template vs full channel data cross-correlation global distribution (reference Figure 2 style)"""
    window_samples = int(window_ms * fs / 1000)
    corr = correlate(channel_data, template, mode='valid')  # Calculate full channel cross-correlation
    corr_norm = corr / (np.std(channel_data) * np.std(template) * window_samples + 1e-8)
    corr_norm /= np.max(np.abs(corr_norm)) if np.max(np.abs(corr_norm)) != 0 else 1
    
    time_axis = np.arange(len(corr_norm)) / fs  # Global time axis (seconds)
    above_thresh = corr_norm > adaptive_thresh  # Points exceeding adaptive threshold
    
    # Plot twin subplots
    fig, (ax_raw, ax_corr) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f'Patient {patient_id} - Template vs Channel Correlation\n'
        f'Window: {window_ms}ms | Adaptive Threshold: {adaptive_thresh:.3f}',
        fontsize=14, y=1.02
    )
    
    # Upper plot: Raw EEG data (first 10 seconds)
    plot_duration = min(10, len(channel_data)/fs)
    plot_samples = int(plot_duration * fs)
    ax_raw.plot(np.arange(plot_samples)/fs, channel_data[:plot_samples], color='blue', linewidth=1)
    ax_raw.set_title('Raw Data (First 10.0s) - Channel', fontsize=12)
    ax_raw.set_ylabel('Amplitude', fontsize=10)
    ax_raw.grid(linestyle=':', alpha=0.3)
    
    # Lower plot: Cross-correlation distribution + adaptive threshold
    ax_corr.plot(time_axis[:plot_samples], corr_norm[:plot_samples], color='orange', linewidth=1.5, label='Normalized Correlation')
    ax_corr.axhline(y=adaptive_thresh, color='red', linestyle='--', linewidth=1, label=f'Adaptive Threshold ({adaptive_thresh:.3f})')
    ax_corr.scatter(time_axis[above_thresh][:plot_samples], corr_norm[above_thresh][:plot_samples], 
                    color='green', s=20, alpha=0.6, label='Above Adaptive Threshold')
    ax_corr.set_title('Template Correlation Distribution', fontsize=12)
    ax_corr.set_ylabel('Correlation', fontsize=10)
    ax_corr.set_xlabel('Time (s)', fontsize=10)
    ax_corr.set_xlim(0, plot_duration)
    ax_corr.grid(linestyle=':', alpha=0.3)
    ax_corr.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    plt.close()

# ==============================================================================
# 9. Main Program (unchanged, automatically pass patient ID when calling)
# ==============================================================================
def main():
    # Step 1: Load data
    print("="*60)
    print("Step 1: Loading EEG Data")
    print("="*60)
    patient_list = load_and_preprocess_data(
        DATA_CONFIG["meta_path"], DATA_CONFIG["data_dir"],
        DATA_CONFIG["sfreq"], DATA_CONFIG["channel_names"]
    )
    if not patient_list:
        print("No valid patient data, program terminated")
        return
    
    # Step 2: Detect candidate spikes (logic unchanged)
    print("="*60)
    print("Step 2: Chatrian Template Matching for Candidate Spike Detection")
    print("="*60)
    for i, patient in enumerate(patient_list):
        print(f"Processing patient {patient['pid']} ({i+1}/{len(patient_list)})")
        candidates, bp_signals, av_bp_valid = detect_candidate_spikes(
            patient, METHOD_CONFIG, CHANNEL_MAP
        )
        patient_list[i]["template_candidates"] = candidates
        patient_list[i]["bp_signals"] = bp_signals
        patient_list[i]["av_bp_valid"] = av_bp_valid
        
        # Plot global cross-correlation chart (logic unchanged)
        core_av = CHANNEL_MAP["core_av"][0]
        if core_av not in patient["channel_idx"]:
            core_av = CHANNEL_MAP["core_av"][1]
        core_ch_idx = patient["channel_idx"][core_av]
        core_ch_data = patient["raw_data"][core_ch_idx]
        chatrian_template, _ = create_chatrian_template(
            fs=DATA_CONFIG["sfreq"],
            window_ms=METHOD_CONFIG["chatrian_window_ms"],
            raw_data=core_ch_data,
            target_amplitude_uv=METHOD_CONFIG["chatrian_target_amp_uv"]
        )
        
        print(f"📊 Plotting global cross-correlation distribution for patient {patient['pid']}...")
        plot_template_correlation(
            template=chatrian_template,
            channel_data=core_ch_data,
            fs=DATA_CONFIG["sfreq"],
            window_ms=METHOD_CONFIG["chatrian_window_ms"],
            patient_id=patient["pid"],
            adaptive_thresh=METHOD_CONFIG["corr_threshold"]
        )
    
    # Step 2.5: FPS validation (logic unchanged)
    print("="*60)
    print("Step 2.5: FPS Bipolar Montage Validation of Candidate Spikes")
    print("="*60)
    for i, patient in enumerate(patient_list):
        pid = patient["pid"]
        print(f"Performing FPS validation for patient {pid}...")
        template_candidates = patient["template_candidates"]
        if not template_candidates:
            print(f"Patient {pid} has no template-matched candidates, skipping FPS validation")
            continue
        
        valid_candidates, bp_channels_fps = fps_validate_candidates(
            raw_data=patient["raw_data"],
            channel_idx=patient["channel_idx"],
            merged_candidates=template_candidates,
            fs=DATA_CONFIG["sfreq"],
            spike_window_ms=METHOD_CONFIG["chatrian_window_ms"],
            train_samples=patient["train_samples"]
        )
        patient_list[i]["valid_candidates"] = valid_candidates
        print(f"Patient {pid} valid candidate count after FPS validation: {len(valid_candidates)} (removed {len(template_candidates)-len(valid_candidates)} invalid candidates)\n")
    
    # Step 3: Feature extraction and label annotation (logic unchanged, debugging logs already added in functions)
    print("="*60)
    print("Step 3: Feature Extraction and Label Annotation")
    print("="*60)
    try:
        X_balanced, y_balanced = create_feature_label(patient_list, METHOD_CONFIG)
    except ValueError as e:
        print(f"Feature extraction failed: {e}")
        return
    
    # Step 4: Train ANN model (logic unchanged)
    print("="*60)
    print("Step 4: Training ANN Classification Model")
    print("="*60)
    model, metrics = train_ann_model(X_balanced, y_balanced, METHOD_CONFIG)
    print(f"\nModel training completed, best model saved to 'best_spike_ann.pth'")

if __name__ == "__main__":
    main()