import os
import numpy as np
import pandas as pd
import mne
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import math
from collections import defaultdict

# === 1. Configuration Parameters ===
DATA_CONFIG = {
    "meta_path": "path/to/Train model/metadata/spick_time_label.xlsx",  # Path to metadata file
    "data_dir": "path/to/Train model/data",  # EEG data directory
    "sfreq": 500,  # Sampling frequency
    "n_total_channels": 16,
    "channel_names": ['FP1','FP2','F3','F4','C3','C4','P3','P4', 
                      'O1','O2','F7','F8','T3','T4','T5','T6'],
    "spike_window_ms": 200,  # Template and segment length
    "cross_corr_thresh": 0.85,
    "cluster_min_ratio": 0.05,
    "merge_interval_ms": 50,
    "rf_n_estimators": 100,
    "train_test_split_ratio": 0.5,
    "bg_sample_count": 100,
    "primary_channels": ["C3", "T3", "C4", "T4"],  # Primary channel list
    "n_initial_templates": 3  # Number of templates to extract per primary channel
}

# Brain region definitions (including primary channels)
BRAIN_REGIONS = {
    "R1": {"channels": ["F3", "C3", "P3"], "primary": "C3"},    # Left central region
    "R2": {"channels": ["F4", "C4", "P4"], "primary": "C4"},    # Right central region
    "R3": {"channels": ["F7", "T3", "T5"], "primary": "T3"},    # Left temporal region
    "R4": {"channels": ["F8", "T4", "T6"], "primary": "T4"}     # Right temporal region
}

# Monopolar to bipolar montage mapping
AV_TO_BP_MAP = {
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

# === 2. Utility Functions ===
def calculate_channel_energy(signal):
    """Calculate channel energy"""
    return np.sum(signal **2, axis=-1)

def create_chatrian_template(fs, window_ms, raw_data, target_amplitude_uv=200):
    """Create Chatrian template (backup)"""
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

def extract_primary_channel_templates(raw_data_0, true_spike_indices, channel_map, target_ch, fs, window_ms=200, n_clusters=3):
    """Extract primary channel templates from true spike segments"""
    raw_data = raw_data_0
    primary_templates = []
    window_samples = int(window_ms * fs / 1000)
    half_window = window_samples // 2
    t = np.linspace(-window_ms/2, window_ms/2, window_samples)

    if target_ch not in channel_map:
        print(f"Warning: Target primary channel {target_ch} not in data")
        return primary_templates
    ch_idx = channel_map[target_ch]

    # Extract valid true spike segments
    true_spike_segments = []
    for true_idx in true_spike_indices:
        start = max(0, true_idx - half_window)
        end = min(raw_data.shape[1], true_idx + half_window)
        ch_segment = raw_data[ch_idx, start:end].copy()

        if len(ch_segment) < window_samples:
            pad_left = (window_samples - len(ch_segment)) // 2
            pad_right = window_samples - len(ch_segment) - pad_left
            ch_segment = np.pad(ch_segment, (pad_left, pad_right), mode='constant')
        true_spike_segments.append(ch_segment)

    if len(true_spike_segments) == 0:
        print(f"Warning: Target primary channel {target_ch} has no valid true labels, cannot extract templates")
        return primary_templates
    print(f"Target primary channel {target_ch}: extracted {len(true_spike_segments)} valid true spikes")

    # Normalization, alignment, and polarity calibration (force peak upward)
    normalized_segments = []
    for seg in true_spike_segments:
        # 1. Normalization
        seg_norm = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
        # 2. Find peak position
        peak_idx = np.argmax(np.abs(seg_norm))  # Peak index (regardless of polarity)
        # 3. Force peak upward: if peak is negative, invert entire segment
        if seg_norm[peak_idx] < 0:
            seg_norm = -seg_norm  # Invert polarity to make peak positive (upward)
        # 4. Align peak to window center
        shift = half_window - peak_idx
        seg_aligned = np.roll(seg_norm, shift)
        normalized_segments.append(seg_aligned)

    # Adaptive number of clusters
    n_clusters = min(n_clusters, len(normalized_segments), 5)
    n_clusters = max(n_clusters, 1)

    # KMeans clustering (all segments now have upward peaks)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(normalized_segments)

    # Generate templates (all with upward peaks)
    for cluster_id in range(n_clusters):
        cluster_center = kmeans.cluster_centers_[cluster_id]
        n_members = np.sum(kmeans.labels_ == cluster_id)

        primary_templates.append({
            "segment": cluster_center,          # Cluster center (peak upward)
            "channel": target_ch,               # Target channel
            "cluster_id": cluster_id,           # Cluster ID
            "n_members": n_members,             # Number of samples in cluster
            "time_axis": t                      # Time axis
        })

    print(f"Target primary channel {target_ch}: generated {len(primary_templates)} initial templates with upward peaks")
    return primary_templates

def correlate(x, y, mode='valid'):
    """
    Custom cross-correlation function, only supports 'mode=valid' (consistent with scipy.signal.correlate logic)
    
    Parameters:
        x: Input signal (1D numpy array), length must ≥ template y length
        y: Template (1D numpy array), used as sliding window matching template
        mode: Only supports 'valid', returns only fully matched window results (no boundary padding)
    
    Returns:
        corr_result: Cross-correlation result (1D numpy array), length = len(x) - len(y) + 1
    """
    # Only support valid mode (matches your code scenario)
    if mode != 'valid':
        raise ValueError("Currently only supports mode='valid', other modes need additional extension")
    
    # Get signal and template lengths
    len_x = len(x)
    len_y = len(y)
    
    # Check if signal length is sufficient (must accommodate at least one template window)
    if len_x < len_y:
        raise ValueError(f"Signal length ({len_x}) must ≥ template length ({len_y})")
    
    # Calculate cross-correlation result length (number of valid windows)
    corr_len = len_x - len_y + 1
    # Initialize cross-correlation result array
    corr_result = np.zeros(corr_len, dtype=np.float64)
    
    # Core: sliding window cross-correlation calculation (consistent with scipy logic)
    # Iterate over each valid window starting position
    for i in range(corr_len):
        # 1. Extract current window from signal x (length = template length)
        x_window = x[i:i + len_y]
        # 2. Calculate cross-correlation between current window and template y (element-wise multiplication then sum)
        # Note: This is "uncentered cross-correlation", consistent with scipy default behavior
        corr_value = np.sum(x_window/(np.max(x_window)-np.min(x_window)) * y)
        # 3. Save current window's cross-correlation result
        corr_result[i] = corr_value
    
    return corr_result

def plot_template_correlation(templates, channel_data, channel_name, fs, window_ms=200, patient_id="Unknown", save_path=None):
    """
    Plot cross-correlation results between initial templates and corresponding channel data,
    using adaptive threshold to mark points exceeding threshold
    
    templates: List of initial templates
    channel_data: Raw data of corresponding channel
    channel_name: Channel name
    fs: Sampling frequency
    window_ms: Window length
    """
    if not templates:
        print("Warning: No templates available to plot correlation results")
        return
    
    window_samples = int(window_ms * fs / 1000)
    time_total = np.arange(len(channel_data)) / fs  # Total time axis (seconds)
    
    # ===================== Added: Calculate adaptive threshold (consistent with detect_candidate_peaks logic) =====================
    # 1. Background noise adjustment (paper Section II.C: using EEG background noise morphology parameters)
    bg_samples = int(1 * fs)  # Use first 1 second signal to estimate background noise (standard EEG preprocessing)
    if len(channel_data) < bg_samples:
        bg_std = np.std(channel_data)  # Signal too short, use global standard deviation
    else:
        bg_std = np.std(channel_data[:bg_samples])  # First 1 second background noise standard deviation
    signal_global_std = np.std(channel_data)
    noise_ratio = bg_std / (signal_global_std + 1e-8)  # Noise to signal amplitude ratio (0~1)
    
    # Threshold adjustment coefficient: higher noise → smaller coefficient (minimum 0.7, avoid threshold too low causing false positives)
    noise_adjust_coeff = 1.0 - min(0.2, noise_ratio * 0.6)  
    adaptive_thresh = DATA_CONFIG["cross_corr_thresh"] * noise_adjust_coeff  # Initial adaptive threshold

    # 2. Cross-correlation distribution safety net (avoid extreme values, paper Section II.C: using cross-correlation morphology distribution)
    # First calculate one template's cross-correlation for distribution statistics (ensure sufficient sample size)
    first_template = templates[0]["segment"]
    first_corr = correlate(channel_data, first_template, mode='valid')
    first_corr_norm = first_corr / (np.std(channel_data) * np.std(first_template) * len(first_template) + 1e-8)
    first_corr_norm /= np.max(np.abs(first_corr_norm)) if np.max(np.abs(first_corr_norm)) != 0 else 1
    
    if len(first_corr_norm) > 10:  # Only calculate distribution when enough candidates (avoid bias from insufficient samples)
        # Take 70th percentile of positive cross-correlation (paper implied logic: cover most true candidates, filter low correlation artifacts)
        corr_quantile = np.quantile(first_corr_norm[first_corr_norm > 0], 0.7)
        adaptive_thresh = max(adaptive_thresh, corr_quantile)  # Threshold not lower than quantile

    # Print adaptive threshold results (consistent with detection phase for traceability)
    print(f"Channel {channel_name} (plotting): fixed threshold {DATA_CONFIG['cross_corr_thresh']} → adaptive {adaptive_thresh:.3f} (noise ratio {noise_ratio:.2f})")
    # =========================================================================
    
    # Calculate cross-correlation for each template with channel data (retain original logic)
    corr_results = []
    for temp in templates:
        template = temp["segment"]
        # Calculate cross-correlation
        corr = correlate(channel_data, template, mode='valid')
        # Normalization (identical to detect_candidate_peaks)
        corr_norm = corr / (np.std(channel_data) * np.std(template) * len(template) + 1e-8)  # Pearson correlation normalization
        corr_norm /= np.max(np.abs(corr_norm)) if np.max(np.abs(corr_norm)) != 0 else 1  # Relative normalization (scaled to ±1)
        # Calculate cross-correlation result time axis
        corr_time = (np.arange(len(corr_norm)) + window_samples // 2) / fs
        corr_results.append({
            "corr_norm": corr_norm,
            "corr_time": corr_time,
            "template": template,
            "cluster_id": temp["cluster_id"],
            "n_members": temp["n_members"]
        })
    
    # Create subplots (retain original logic)
    n_templates = len(templates)
    fig, axes = plt.subplots(n_templates + 1, 1, figsize=(12, 3*(n_templates + 1)), sharex=True)
    fig.suptitle(
        f'Patient {patient_id} - Template vs Channel {channel_name} Correlation\n'
        f'Window: {window_ms}ms | Adaptive Threshold: {adaptive_thresh:.3f}',  # Title includes adaptive threshold
        fontsize=14, y=1.02
    )
    
    # Plot raw channel data (first 10 seconds, retain original logic)
    plot_duration = min(10, len(channel_data)/fs)  # Show up to 10 seconds
    plot_samples = int(plot_duration * fs)
    ax_raw = axes[0]
    ax_raw.plot(time_total[:plot_samples], channel_data[:plot_samples], color='blue', linewidth=1)
    ax_raw.set_title(f'Raw Data (First {plot_duration:.1f}s) - Channel {channel_name}', fontsize=12)
    ax_raw.set_ylabel('Amplitude', fontsize=10)
    ax_raw.grid(linestyle=':', alpha=0.3)
    
    # Plot cross-correlation results for each template (key modification: use adaptive threshold instead of fixed threshold)
    for i, corr_data in enumerate(corr_results, 1):
        ax = axes[i]
        # Plot cross-correlation curve (retain original logic)
        ax.plot(corr_data["corr_time"][:plot_samples], corr_data["corr_norm"][:plot_samples], 
                color='orange', linewidth=1.5, label='Normalized Correlation')
        
        # Key modification 1: Plot adaptive threshold line (replace original fixed threshold)
        ax.axhline(y=adaptive_thresh, color='red', linestyle='--', linewidth=1, 
                   label=f'Adaptive Threshold ({adaptive_thresh:.3f})')
        
        # Key modification 2: Use adaptive threshold to mark points exceeding threshold
        above_threshold = corr_data["corr_norm"] > adaptive_thresh
        ax.scatter(corr_data["corr_time"][above_threshold][:plot_samples], 
                   corr_data["corr_norm"][above_threshold][:plot_samples],
                   color='green', s=20, alpha=0.6, label='Above Adaptive Threshold')
        
        # Retain original title and label logic
        ax.set_title(f'Template Cluster {corr_data["cluster_id"]} (Samples: {corr_data["n_members"]})', fontsize=12)
        ax.set_ylabel('Correlation', fontsize=10)
        ax.grid(linestyle=':', alpha=0.3)
        if i == 1:  # Only show legend in first correlation plot
            ax.legend(fontsize=9)
    
    # Set x-axis label (retain original logic)
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    axes[-1].set_xlim(0, plot_duration)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_eeg_with_candidates(channel_data, candidate_indices, fs, channel_name, pid, save_path=None):
    """Plot EEG signal with candidate spike markers"""
    samples_10s = int(fs * 10)
    channel_data_10s = channel_data[:samples_10s]
    time_10s = np.arange(len(channel_data_10s)) / fs
    
    plt.figure(figsize=(10, 2))
    print(f"Channel {channel_name} (10s): max={np.max(channel_data_10s):.6f}V, mean={np.mean(channel_data_10s):.6f}V")
    plt.plot(time_10s, channel_data_10s, color='blue', linewidth=1)
    
    candidate_indices_10s = [idx for idx in candidate_indices if 0 <= idx < samples_10s]
    for idx in candidate_indices_10s:
        plt.scatter(time_10s[idx], channel_data_10s[idx], color='green', marker='o', s=50, alpha=0.7)
    
    plt.title(f'Patient {pid} - Channel {channel_name} (First 10s) with Candidate Spikes')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (V)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def detect_candidate_peaks(signal, template, fs, cross_corr_thresh=0.7, keep_negative=False, template_info=None):
    """Detect candidate spike peaks using template matching with adaptive threshold"""
    window_samples = len(template)
    n_times = signal.shape[-1]
    candidate_peaks = []
    
    # ===================== Added: Calculate adaptive threshold (completely consistent with plot_template_correlation logic) =====================
    # 1. Background noise adjustment (avoid using first 1 second, use "full signal sliding window" to select spike-free stable segments)
    bg_duration_sec = 1  # Background segment length (1 second, consistent with plotting function)
    bg_samples = int(bg_duration_sec * fs)
    # Calculate sliding window standard deviation of full signal (find low-variance stable segments as background)
    window_size = int(0.5 * fs)  # 0.5 second sliding window
    if len(signal) < window_size:
        bg_std = np.std(signal)
    else:
        # Calculate local standard deviation using sliding window (reflects signal stability)
        rolling_std = np.convolve(
            np.square(signal), 
            np.ones(window_size)/window_size, 
            mode='valid'
        )
        # Find 1-second segment with minimum standard deviation (most stable, likely pure background)
        min_std_idx = np.argmin(rolling_std[:-bg_samples + window_size])
        bg_data = signal[min_std_idx : min_std_idx + bg_samples]
        bg_std = np.std(bg_data)
    
    signal_global_std = np.std(signal)
    noise_ratio = bg_std / (signal_global_std + 1e-8)  # Noise ratio (0~1)
    # Threshold adjustment coefficient: higher noise → smaller coefficient (minimum 0.8, stricter than plotting to reduce false positives)
    noise_adjust_coeff = 1.0 - min(0.2, noise_ratio * 0.6)  # Consistent with your plotting function's 0.2
    adaptive_thresh = cross_corr_thresh * noise_adjust_coeff  # Adaptive threshold

    # 2. Cross-correlation distribution safety net (avoid extreme values)
    # First calculate cross-correlation for distribution statistics
    temp_corr = correlate(signal, template, mode='valid')
    temp_corr_norm = temp_corr / (np.std(signal) * np.std(template) * len(template) + 1e-8)
    temp_corr_norm /= np.max(np.abs(temp_corr_norm)) if np.max(np.abs(temp_corr_norm)) != 0 else 1
    # When enough candidates, use 70th percentile as safety net (avoid threshold too low)
    if len(temp_corr_norm) > 10:
        corr_quantile = np.quantile(temp_corr_norm[temp_corr_norm > 0], 0.7)
        adaptive_thresh = max(adaptive_thresh, corr_quantile)

    # Print adaptive threshold results (for debugging, format consistent with plotting function)
    if template_info:
        channel = template_info["channel"]
        print(f"Channel {channel} (detection): fixed threshold {cross_corr_thresh} → adaptive {adaptive_thresh:.3f} (noise ratio {noise_ratio:.2f})")
    # =========================================================================
    
    # 1. Signal scaling: consistent with plot function
    scaled_signal = signal  # Uniform scaling logic unchanged
    
    # 2. Calculate cross-correlation (unnormalized)
    cross_corr = correlate(scaled_signal, template, mode='valid')
    
    # 3. Normalization: completely consistent with plot function
    cross_corr_norm = cross_corr / (np.std(signal) * np.std(template) * len(template)) + 1e-8
    cross_corr_norm /= np.max(np.abs(cross_corr_norm)) if np.max(np.abs(cross_corr_norm)) != 0 else 1
    
    # 4. Key modification: Use adaptive threshold to filter candidates (replace original fixed threshold)
    peak_indices_pos = np.where(cross_corr_norm > adaptive_thresh)[0]
    
    # 5. Candidate spike verification (original logic unchanged)
    for idx in peak_indices_pos:
        peak_center = idx + window_samples // 2
        spike_segment = signal[idx:idx+window_samples].copy()
        
        # Handle boundary padding
        if len(spike_segment) < window_samples:
            pad_left = (window_samples - len(spike_segment)) // 2
            pad_right = window_samples - len(spike_segment) - pad_left
            spike_segment = np.pad(spike_segment, (pad_left, pad_right), mode='constant')
        
        # Amplitude verification
        background_std = np.std(spike_segment)
        peak_amplitude = np.max(np.abs(spike_segment))
        
        # Polarity verification (only keep upward peaks, consistent with template)
        seg_peak_idx = np.argmax(np.abs(spike_segment))
        if spike_segment[seg_peak_idx] < 0:
            continue
        
        # Amplitude threshold verification
        if peak_amplitude > 2 * background_std:
            candidate_peaks.append({
                "center_idx": peak_center,
                "cross_corr": cross_corr_norm[idx],
                "amplitude": peak_amplitude,
                "segment": spike_segment,
                "polarity": "positive",
                "adaptive_thresh": adaptive_thresh  # Record threshold for subsequent analysis
            })
    
    # Print detection results
    if template_info:
        cluster_id = template_info["cluster_id"]
        channel = template_info["channel"]
        print(f"Template cluster {cluster_id} (channel {channel}): detected {len(candidate_peaks)} upward peak candidate spikes")
    else:
        print(f"Detected {len(candidate_peaks)} upward peak candidate spikes")
    
    if keep_negative:
        print("Warning: Forced to keep only upward peak candidates, negative polarity candidates not retained")
    
    return candidate_peaks

def plot_dynamic_grid(spike_list, titles, fig_title, fs, window_ms=200, save_path=None, max_cols=3, template_info_list=None):
    """Plot multiple spike candidates in a dynamic grid layout"""
    n_spikes = len(spike_list)
    if n_spikes == 0:
        print("No spike data to plot")
        return
    
    if template_info_list is not None and len(template_info_list) != n_spikes:
        print("Warning: Template list length doesn't match candidate list, not plotting templates")
        template_info_list = None
    
    cols = min(max_cols, n_spikes)
    rows = math.ceil(n_spikes / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    fig.suptitle(fig_title, fontsize=16, y=1.02)
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    window_samples = int(window_ms * fs / 1000)
    
    for i in range(n_spikes):
        row_idx = i // cols
        col_idx = i % cols
        ax = axes[row_idx, col_idx]
        
        spike = spike_list[i]
        segment = spike.get("segment", None)
        if segment is None:
            signal = spike["signal"]
            center_idx = spike["center_idx"]
            start_idx = max(0, center_idx - window_samples // 2)
            end_idx = min(len(signal), center_idx + window_samples // 2)
            segment = signal[start_idx:end_idx]
        
        center_idx = spike["center_idx"]
        cross_corr = spike.get("cross_corr", 0)
        ch_name = spike.get("channel", "Unknown")
        
        t_relative = np.arange(len(segment)) / fs - (len(segment) // 2) / fs
        
        ax.plot(t_relative, segment, color='blue', linewidth=1.5, label='Candidate Spike')
        ax.scatter(0, segment[len(segment) // 2], color='red', s=50, marker='*', label='Peak')
        
        if template_info_list is not None:
            template_info = template_info_list[i]
            template = template_info["template"]
            template_ch = template_info.get("channel", "Unknown")
            
            if template_ch == ch_name or template_ch == "Unknown":
                if len(template) != len(segment):
                    pad_left = (len(segment) - len(template)) // 2
                    pad_right = len(segment) - len(template) - pad_left
                    template = np.pad(template, (pad_left, pad_right), mode='constant')
                
                segment_max = np.max(np.abs(segment))
                if segment_max != 0:
                    template = template * (segment_max * 0.8)
                
                ax.plot(t_relative, template, color='orange', linewidth=1.2, 
                        linestyle='--', label='Matching Template')
        
        ax.set_title(titles[i], fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("Amplitude", fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', labelsize=6)
        if i == 0:
            ax.legend(fontsize=6, loc='upper right')
        
        ax.text(0.05, 0.92, f"Corr: {cross_corr:.2f}", 
                transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    for i in range(n_spikes, rows*cols):
        row_idx = i // cols
        col_idx = i % cols
        axes[row_idx, col_idx].axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def normalize_and_align(waveform):
    """Normalize and align waveform to center"""
    waveform = (waveform - np.mean(waveform)) / (np.std(waveform) + 1e-8)
    peak_idx = np.argmax(np.abs(waveform))
    half_len = len(waveform) // 2
    shift = half_len - peak_idx
    return np.roll(waveform, shift)

def kmeans_cluster_candidates(candidate_peaks, n_clusters=3):
    """Cluster candidate spikes using KMeans"""
    cluster_groups = []
    channel_to_peaks = defaultdict(list)
    
    for peak in candidate_peaks:
        if "channel" in peak:
            channel_to_peaks[peak["channel"]].append(peak)
    
    for ch, ch_peaks in channel_to_peaks.items():
        if len(ch_peaks) < n_clusters:
            print(f"Channel {ch} candidate spike count ({len(ch_peaks)}) < cluster count ({n_clusters}), skipping clustering")
            continue
        
        peak_waveforms = []
        valid_peaks = []
        
        for peak in ch_peaks:
            waveform = peak.get("segment")
            if waveform is None:
                print("Warning: Candidate spike missing segment data, skipping")
                continue
                
            if len(peak_waveforms) > 0 and len(waveform) != len(peak_waveforms[0]):
                target_len = len(peak_waveforms[0])
                pad_left = (target_len - len(waveform)) // 2
                pad_right = target_len - len(waveform) - pad_left
                waveform = np.pad(waveform, (pad_left, pad_right), mode='constant')
            
            waveform = normalize_and_align(waveform)
            
            peak_waveforms.append(waveform)
            valid_peaks.append(peak)
        
        if len(peak_waveforms) < n_clusters:
            print(f"Channel {ch} insufficient valid segments, skipping clustering")
            continue
        
        peak_features = np.array(peak_waveforms)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(peak_features)
        
        for cluster_id in range(n_clusters):
            cluster_mask = (clusters == cluster_id)
            cluster_peaks = [valid_peaks[i] for i in np.where(cluster_mask)[0]]
            if not cluster_peaks:
                continue
            
            cluster_center = int(np.mean([p["center_idx"] for p in cluster_peaks]))
            
            cluster_groups.append({
                "cluster_id": f"{ch}_{cluster_id}",
                "channel": ch,
                "peaks": cluster_peaks,
                "center_idx": cluster_center,
                "size": len(cluster_peaks),
                "centroid": kmeans.cluster_centers_[cluster_id]
            })
    
    return cluster_groups

def raw_cluster(cluster_groups, total_candidates, min_ratio=0.05):
    """Filter clusters based on minimum size ratio"""
    min_size = total_candidates * min_ratio
    return [g for g in cluster_groups if g["size"] >= min_size]

def create_adaptive_templates(cluster_groups, fs, window_ms=100):
    """Create adaptive templates from clustered spike waveforms"""
    adaptive_templates = []
    window_samples = int(window_ms * fs / 1000)
    
    for cluster in cluster_groups:
        template = cluster["centroid"].copy()
        
        # Adjust template length
        if len(template) < window_samples:
            pad_left = (window_samples - len(template)) // 2
            pad_right = window_samples - len(template) - pad_left
            template = np.pad(template, (pad_left, pad_right), mode='constant')
        elif len(template) > window_samples:
            template = template[:window_samples]
        
        # Polarity calibration: force peak upward
        peak_idx = np.argmax(np.abs(template))
        if template[peak_idx] < 0:
            template = -template  # Invert polarity to ensure positive peak
        
        # Normalization
        template = (template - np.mean(template)) / (np.std(template) + 1e-8)
        
        adaptive_templates.append({
            "cluster_id": cluster["cluster_id"],
            "channel": cluster["channel"],
            "template": template,                # Final template (peak upward)
            "center_idx": cluster["center_idx"]
        })
    print(f"Generated {len(adaptive_templates)} adaptive templates with upward peaks")
    return adaptive_templates

def plot_adaptive_template(template_info, fs, window_ms=100, save_path=None):
    """Plot adaptive template waveform"""
    template = template_info["template"]
    cluster_id = template_info["cluster_id"]
    channel = template_info["channel"]
    center_idx = template_info["center_idx"]
    
    window_samples = len(template)
    t = np.linspace(-window_ms/2, window_ms/2, window_samples)
    t_sec = t / 1000
    
    plt.figure(figsize=(8, 4))
    plt.plot(t_sec, template, color='red', linewidth=2)
    plt.title(f"Adaptive Template - Cluster: {cluster_id}, Channel: {channel}, Center Index: {center_idx}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def merge_overlapping_peaks(candidate_peaks, merge_interval_ms, fs):
    """Merge overlapping candidate peaks"""
    if len(candidate_peaks) == 0:
        return []
    
    sorted_peaks = sorted(candidate_peaks, key=lambda x: x["center_idx"])
    merged_peaks = [sorted_peaks[0]]
    merge_interval_samples = int(merge_interval_ms * fs / 1000)
    
    for peak in sorted_peaks[1:]:
        last_peak = merged_peaks[-1]
        if peak["center_idx"] - last_peak["center_idx"] < merge_interval_samples:
            if peak["amplitude"] > last_peak["amplitude"]:
                merged_peaks[-1] = peak
        else:
            merged_peaks.append(peak)
    return merged_peaks

def calculate_bp_peak_to_peak(bp_signal, peak_center_idx, fs, window_ms=100):
    """Calculate peak-to-peak amplitude in bipolar montage signal"""
    window_samples = int(window_ms * fs / 1000)
    start_idx = max(0, peak_center_idx - window_samples // 2)
    end_idx = min(bp_signal.shape[-1], peak_center_idx + window_samples // 2)
    window_signal = bp_signal[start_idx:end_idx]
    return np.max(window_signal) - np.min(window_signal)

def extract_spike_features(signal_window, fs):
    """Extract time-domain features from spike waveform"""
    window_len = len(signal_window)
    t = np.arange(window_len) / fs * 1000
    signal_abs = np.abs(signal_window)
    peak_idx = np.argmax(signal_abs)
    peak_10pct = 0.1 * signal_abs[peak_idx]

    left_mask = signal_window[:peak_idx] < peak_10pct
    left_bound_idx = np.where(left_mask)[0][0] if np.any(left_mask) else 0
    dur_left = t[peak_idx] - t[left_bound_idx]
    
    right_mask = signal_window[peak_idx:] < peak_10pct
    right_bound_idx = (np.where(right_mask)[0][-1] + peak_idx) if np.any(right_mask) else (window_len - 1)
    dur_right = t[right_bound_idx] - t[peak_idx]
    dur_spike = dur_left + dur_right

    amp_left = np.max(np.abs(signal_window[:peak_idx])) if peak_idx > 0 else 0.0
    amp_right = np.max(np.abs(signal_window[peak_idx+1:])) if peak_idx < window_len-1 else 0.0
    amp_spike = signal_abs[peak_idx]

    max_left_step = min(5, peak_idx)
    slope_left = (signal_window[peak_idx] - signal_window[peak_idx - max_left_step]) / (max_left_step / fs * 1000) if max_left_step > 0 else 0.0
    
    max_right_step = min(5, window_len - 1 - peak_idx)
    slope_right = (signal_window[peak_idx + max_right_step] - signal_window[peak_idx]) / (max_right_step / fs * 1000) if max_right_step > 0 else 0.0
    slope_spike = slope_left - slope_right

    area_spike = simpson(y=np.abs(signal_window), x=t)

    return [
        dur_left, dur_right, dur_spike,
        amp_left, amp_right, amp_spike,
        slope_left, slope_right, slope_spike,
        area_spike
    ]

def calculate_paper_metrics(TP, FP, FN, TN):
    """Calculate performance metrics (Sensitivity, Specificity, Selectivity, Accuracy)"""
    Sen = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0.0
    Spe = TN / (TN + FP) * 100 if (TN + FP) > 0 else 0.0
    Sel = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0.0
    AC = (TP + TN) / (TP + TN + FP + FN) * 100 if (TP + TN + FP + FN) > 0 else 0.0
    return {"Sen": round(Sen, 1), "Spe": round(Spe, 1), "Sel": round(Sel, 1), "AC": round(AC, 1)}

def load_and_preprocess_data(config):
    """Load and preprocess patient EEG data"""
    meta_df = pd.read_excel(config["meta_path"], header=None)
    patient_list = []
    
    for idx, row in meta_df.iterrows():
        pid_full = row[0].strip()
        pid = pid_full[:11]
        eeg_path = os.path.join(config["data_dir"], f"{pid_full}.set")
        
        if not os.path.exists(eeg_path):
            print(f"Warning: Patient {pid} EEG file doesn't exist, skipping")
            continue
        
        try:
            raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
            raw_data = raw.get_data()
            print("max(raw_data)", np.max(raw_data))
        except Exception as e:
            print(f"Warning: Patient {pid} EEG reading failed: {str(e)}, skipping")
            continue
        
        if raw.ch_names != config["channel_names"]:
            print(f"Warning: Patient {pid} channel order mismatch (actual: {raw.ch_names[:5]}...), skipping")
            continue
        
        true_spike_times_sec = row[1:].dropna().astype(float).tolist()
        if not true_spike_times_sec:
            print(f"Warning: Patient {pid} has no true spike annotations, skipping")
            continue
        
        true_spike_indices = [int(t * config["sfreq"]) for t in true_spike_times_sec]
        true_spike_indices = [idx for idx in true_spike_indices if 0 <= idx < raw_data.shape[1]]
        if not true_spike_indices:
            print(f"Warning: Patient {pid} true spikes out of data range, skipping")
            continue
        
        total_samples = raw_data.shape[1]
        train_samples = int(total_samples * config["train_test_split_ratio"])
        train_true_spikes = [idx for idx in true_spike_indices if idx <= train_samples]
        test_true_spikes = [idx for idx in true_spike_indices if idx > train_samples]
        
        print(f"Patient {pid}: total duration={total_samples/config['sfreq']:.1f}s, training set={train_samples/config['sfreq']:.1f}s, total true spikes={len(true_spike_indices)} (training={len(train_true_spikes)}, test={len(test_true_spikes)})")
        
        patient_list.append({
            "pid": pid,
            "raw_data": raw_data,
            "train_samples": train_samples,
            "train_true_spikes": train_true_spikes,
            "test_true_spikes": test_true_spikes,
            "channel_map": {ch: i for i, ch in enumerate(config["channel_names"])}
        })
    
    print(f"\nSuccessfully loaded {len(patient_list)} patient data files\n")
    return patient_list

def train_bect_model(patient_data, config):
    """Train BECT spike detection model"""
    fs = config["sfreq"]
    spike_window_samples = int(config["spike_window_ms"] * fs / 1000)
    channel_map = patient_data["channel_map"]
    raw_data = patient_data["raw_data"]
    train_samples = patient_data["train_samples"]
    train_true_spikes = patient_data["train_true_spikes"]  # Only for label annotation, not for template generation
    pid = patient_data["pid"]
    
    # 1. Region energy analysis to determine target region and corresponding primary channel
    region_energy = {}
    for region_name, region_info in BRAIN_REGIONS.items():
        region_channels = region_info["channels"]
        region_indices = [channel_map[ch] for ch in region_channels if ch in channel_map]
        if not region_indices:
            continue
        region_data = raw_data[region_indices, :train_samples]
        # Within-region normalization (standardize within region channels, preserve relative differences)
        region_mean = np.mean(region_data, axis=1, keepdims=True)
        region_std = np.std(region_data, axis=1, keepdims=True) + 1e-8
        region_data_norm = (region_data - region_mean) / region_std  # Within-region normalization
        channel_energies = calculate_channel_energy(region_data_norm) 
        region_energy[region_name] = np.sum(channel_energies)
    
    if not region_energy:
        print(f"Patient {pid}: No valid region data, training failed")
        return None
    target_region = max(region_energy.keys(), key=lambda k: region_energy[k])
    target_region_info = BRAIN_REGIONS[target_region]
    target_channels = target_region_info["channels"]
    target_primary_channel = target_region_info["primary"]  # Target primary channel (for generating Chatrian template)
    print(f"Patient {pid} (training): target region={target_region}, processing channels={target_channels}, primary channel={target_primary_channel}")

    # 2. Core: Create fixed Chatrian template (replace real spike clustering templates)
    all_candidate_peaks = []
    train_raw_data = raw_data[:, :train_samples]

    # 2.1 Validate primary channel effectiveness and generate template
    if target_primary_channel not in channel_map:
        print(f"Warning: Target primary channel {target_primary_channel} not in data, training failed")
        return None
    ch_idx = channel_map[target_primary_channel]
    ch_data = train_raw_data[ch_idx]  # Use primary channel training data to adapt template amplitude

    # Create fixed Chatrian template
    chatrian_template, _ = create_chatrian_template(
        fs=fs,
        window_ms=config["spike_window_ms"],
        raw_data=ch_data
    )
    print(f"Patient {pid}: Successfully created fixed Chatrian template for {target_primary_channel} channel")

    # 2.3 Use Chatrian template to detect all channels in target region (generate initial candidate spikes)
    print(f"Using Chatrian template to detect all channels in target region {target_region}...")
    for ch_name in target_channels:
        if ch_name not in channel_map:
            print(f"Warning: Channel {ch_name} doesn't exist, skipping")
            continue
        ch_idx = channel_map[ch_name]
        ch_data = train_raw_data[ch_idx]
        
        # Detect candidate spikes
        ch_candidates = detect_candidate_peaks(
            signal=ch_data,
            template=chatrian_template,
            fs=fs,
            cross_corr_thresh=config["cross_corr_thresh"],
            keep_negative=False,
            template_info={"cluster_id": 0, "channel": ch_name}
        )
        
        # Add channel information and aggregate
        for cand in ch_candidates:
            cand["channel"] = ch_name
            cand["channel_idx"] = ch_idx
        all_candidate_peaks.extend(ch_candidates)
        print(f"Channel {ch_name}: detected {len(ch_candidates)} candidate spikes")

    # 2.4 Visualize cross-correlation results between Chatrian template and primary channel
    print(f"Plotting cross-correlation results between Chatrian template and {target_primary_channel} channel...")
    plot_template_correlation(
        templates=[{"segment": chatrian_template, "cluster_id": 0, "n_members": 1, "channel": target_primary_channel}],
        channel_data=train_raw_data[channel_map[target_primary_channel]],
        channel_name=target_primary_channel,
        fs=fs,
        window_ms=config["spike_window_ms"],
        patient_id=pid,
        save_path=f"./plots/{pid}_chatrian_template_correlation.png"
    )

    # 3. Validate initial candidate spike effectiveness
    if not all_candidate_peaks:
        print(f"Patient {pid} (training): No initial candidate spikes, training failed\n")
        return None
    print(f"All channels in target region: detected {len(all_candidate_peaks)} initial candidate spikes")

    # 4. Clustering and adaptive template generation (unchanged logic)
    print("Performing KMeans clustering on initial candidate spikes...")
    cluster_groups = kmeans_cluster_candidates(
        candidate_peaks=all_candidate_peaks,
        n_clusters=2  # Fixed cluster count 2, adjustable as needed
    )
    
    # Filter valid clusters (remove clusters with too few samples)
    raw_clusters = raw_cluster(cluster_groups, len(all_candidate_peaks), config["cluster_min_ratio"])
    if len(raw_clusters) == 0:
        print(f"Patient {pid} (training): No valid clusters, training failed\n")
        return None
    print(f"Clustering results: {len(raw_clusters)} valid clusters (sample count ≥ {config['cluster_min_ratio']*100}% total candidates)")

    # Generate adaptive templates
    adaptive_templates = create_adaptive_templates(
        cluster_groups=raw_clusters,
        fs=fs,
        window_ms=config["spike_window_ms"]
    )
    if not adaptive_templates:
        print(f"Patient {pid} (training): No adaptive templates, training failed\n")
        return None
    print(f"Generated {len(adaptive_templates)} adaptive templates (upward peaks)")

    # 5. Adaptive template detection and verification (unchanged logic)
    print("Re-detecting target region channels with adaptive templates...")
    adaptive_candidates = []
    for template_info in adaptive_templates:
        ch_name = template_info["channel"]
        if ch_name not in channel_map:
            continue
        ch_idx = channel_map[ch_name]
        ch_data = train_raw_data[ch_idx]
        
        # Adaptive template detection
        ch_candidates = detect_candidate_peaks(
            signal=ch_data,
            template=template_info["template"],
            fs=fs,
            cross_corr_thresh=config["cross_corr_thresh"],
            keep_negative=False
        )
        
        # Add channel information and aggregate
        for cand in ch_candidates:
            cand["channel"] = ch_name
            cand["channel_idx"] = ch_idx
        adaptive_candidates.extend(ch_candidates)

    # Merge overlapping candidate spikes (optional: uncomment if deduplication needed)
    merged_candidates = merge_overlapping_peaks(adaptive_candidates, config["merge_interval_ms"], fs)
    merged_candidates = adaptive_candidates
    if len(merged_candidates) == 0:
        print(f"Patient {pid} (training): No candidate spikes after merging, training failed\n")
        return None
    print(f"Adaptive template detection: {len(merged_candidates)} candidate spikes (no merging of overlapping)")

    # Bipolar montage validation (FPS filtering)
    print("Bipolar montage validation (remove invalid candidates)...")
    bp_channels = {}
    # Generate bipolar montage data
    for av_ch1, av_ch2 in [("F3","C3"), ("C3","P3"), ("F4","C4"), ("C4","P4"), 
                           ("F7","T3"), ("T3","T5"), ("F8","T4"), ("T4","T6"),
                           ("FP1","F3"), ("P3","O1"), ("FP2","F4"), ("P4","O2")]:
        bp_name = f"{av_ch1}-{av_ch2}"
        if av_ch1 not in channel_map or av_ch2 not in channel_map:
            continue
        bp_data = train_raw_data[channel_map[av_ch2]] - train_raw_data[channel_map[av_ch1]]
        bp_channels[bp_name] = bp_data

    # Validate bipolar montage peaks for candidate spikes
    valid_candidates = []
    for cand in merged_candidates:
        ch_name = cand["channel"]
        center_idx = cand["center_idx"]
        bp_names = AV_TO_BP_MAP.get(ch_name, [])
        
        if len(bp_names) != 2:
            continue
        bp_valid = []
        for bp_name in bp_names:
            if bp_name not in bp_channels:
                continue
            # Calculate bipolar montage peak-to-peak amplitude
            p2p = calculate_bp_peak_to_peak(bp_channels[bp_name], center_idx, fs, config["spike_window_ms"])
            bg_std = np.std(bp_channels[bp_name])
            bp_valid.append(p2p > 1.5 * bg_std)  # Adjustable threshold (e.g., 0.5*bg_std)
        
        if all(bp_valid):
            valid_candidates.append(cand)

    if len(valid_candidates) == 0:
        print(f"Patient {pid} (training): No valid candidates after FPS filtering, training failed\n")
        return None
    print(f"After FPS validation: valid candidate spikes={len(valid_candidates)} (removed {len(merged_candidates)-len(valid_candidates)} invalid candidates)")

    # 6. Feature extraction and model training (no negative sample supplementation, use original labels)
    print("Extracting features from valid candidate spikes and training RF model...")
    feature_matrix = []
    y_true = []

    # 6.1 Extract features and annotate labels
    for cand in valid_candidates:
        ch_name = cand["channel"]
        center_idx = cand["center_idx"]
        bp_names = AV_TO_BP_MAP.get(ch_name, [])
        
        # Monopolar montage features
        av_data = cand["segment"]
        if len(av_data) != spike_window_samples:
            # Standardize segment length (pad or truncate)
            if len(av_data) < spike_window_samples:
                pad_left = (spike_window_samples - len(av_data)) // 2
                pad_right = spike_window_samples - len(av_data) - pad_left
                av_data = np.pad(av_data, (pad_left, pad_right), mode='constant')
            else:
                av_data = av_data[:spike_window_samples]
        av_features = extract_spike_features(av_data, fs)

        # Bipolar montage features (2 bipolar montages corresponding to monopolar channel)
        bp_features_list = []
        valid_bp = True
        for bp_name in bp_names:
            if bp_name not in bp_channels:
                valid_bp = False
                break
            # Extract bipolar montage data for corresponding window
            start_idx = max(0, center_idx - spike_window_samples // 2)
            end_idx = min(train_samples, center_idx + spike_window_samples // 2)
            bp_data = bp_channels[bp_name][start_idx:end_idx]
            # Standardize bipolar segment length
            if len(bp_data) < spike_window_samples:
                pad_left = (spike_window_samples - len(bp_data)) // 2
                pad_right = spike_window_samples - len(bp_data) - pad_left
                bp_data = np.pad(bp_data, (pad_left, pad_right), mode='constant')
            elif len(bp_data) > spike_window_samples:
                bp_data = bp_data[:spike_window_samples]
            bp_features = extract_spike_features(bp_data, fs)
            bp_features_list.append(bp_features)

        # Only keep samples with complete bipolar features
        if not valid_bp or len(bp_features_list) != 2:
            continue

        # Concatenate monopolar + bipolar features (total 10 + 10 + 10 = 30-dimensional features)
        full_features = av_features + bp_features_list[0] + bp_features_list[1]
        feature_matrix.append(full_features)

        # Annotate sample labels (positive: overlaps with true spike; negative: doesn't overlap)
        is_true_spike = any(
            np.abs(center_idx - true_idx) < spike_window_samples // 2 
            for true_idx in train_true_spikes
        )
        y_true.append(1 if is_true_spike else 0)

    # 6.2 Validate feature sample effectiveness
    if len(feature_matrix) == 0 or len(y_true) == 0:
        print(f"Patient {pid} (training): No valid feature samples, training failed\n")
        return None
    y_true = np.array(y_true)
    n_positive = np.sum(y_true)
    n_negative = len(y_true) - n_positive
    print(f"Feature sample distribution: positive={n_positive} samples, negative={n_negative} samples (total {len(y_true)} samples)")

    # 6.3 Validate completeness of positive/negative samples (need both to train classifier)
    if n_positive == 0 or n_negative == 0:
        print(f"Patient {pid} (training): Missing positive or negative samples, cannot train classifier\n")
        return None

    # 6.4 Train random forest classifier (use original samples, no negative sample supplementation)
    rf_model = RandomForestClassifier(
        n_estimators=config["rf_n_estimators"],
        random_state=42,
        class_weight="balanced",  # Use class weights to mitigate original sample imbalance
        n_jobs=-1  # Multi-threading for faster training
    )
    rf_model.fit(feature_matrix, y_true)
    print(f"Patient {pid} (training): Random forest classifier training completed ({config['rf_n_estimators']} trees)\n")

    # 8. Return training results (for test function use)
    return {
        "pid": pid,
        "adaptive_templates": adaptive_templates,  # Adaptive templates (used for detection during testing)
        "rf_model": rf_model,                      # Trained classifier
        "target_region": target_region,            # Target region (used during testing)
        "target_channels": target_channels,        # Target channels (used during testing)
        "spike_window_samples": spike_window_samples  # Window length (unified for testing)
    }

def test_bect_model(patient_data, train_result, config):
    """Test BECT spike detection model"""
    fs = config["sfreq"]
    spike_window_samples = int(config["spike_window_ms"] * fs / 1000)
    channel_map = patient_data["channel_map"]
    raw_data = patient_data["raw_data"]
    train_samples = patient_data["train_samples"]
    test_true_spikes = patient_data["test_true_spikes"]
    pid = patient_data["pid"]
    
    adaptive_templates = train_result["adaptive_templates"]
    rf_model = train_result["rf_model"]
    target_channels = train_result["target_channels"]
    target_indices = [channel_map[ch] for ch in target_channels if ch in channel_map]
    
    test_raw_data = raw_data[:, train_samples:]
    test_total_samples = test_raw_data.shape[1]
    adaptive_candidates = []
    for template_info in adaptive_templates:
        ch_name = template_info["channel"]
        ch_idx = channel_map[ch_name]
        ch_data = test_raw_data[ch_idx]
        # Normalization
        ch_data = (ch_data - np.mean(ch_data)) / np.std(ch_data)
        ch_candidates = detect_candidate_peaks(
            ch_data, template_info["template"], fs, config["cross_corr_thresh"], keep_negative=False
        )
        for cand in ch_candidates:
            cand["center_idx"] += train_samples
            cand["channel"] = ch_name
            cand["channel_idx"] = ch_idx
        adaptive_candidates.extend(ch_candidates)
    
    merged_candidates = adaptive_candidates
    merged_candidates = merge_overlapping_peaks(adaptive_candidates, config["merge_interval_ms"], fs)
    if len(merged_candidates) == 0:
        print(f"Patient {pid} (test): No candidate spikes, test failed\n")
        return None
    print(f"Patient {pid} (test): Total candidate spikes={len(merged_candidates)}")

    bp_channels = {}
    for av_ch1, av_ch2 in [("F3","C3"), ("C3","P3"), ("F4","C4"), ("C4","P4"), 
                           ("F7","T3"), ("T3","T5"), ("F8","T4"), ("T4","T6"),
                           ("FP1","F3"), ("P3","O1"), ("FP2","F4"), ("P4","O2")]:
        bp_name = f"{av_ch1}-{av_ch2}"
        bp_data = raw_data[channel_map[av_ch2]] - raw_data[channel_map[av_ch1]]
        bp_channels[bp_name] = bp_data
    
    valid_candidates = []
    for cand in merged_candidates:
        ch_name = cand["channel"]
        center_idx = cand["center_idx"]
        bp_names = AV_TO_BP_MAP.get(ch_name, [])
        if len(bp_names) != 2:
            continue
        
        bp_valid = []
        for bp_name in bp_names:
            if bp_name not in bp_channels:
                continue
            bp_data = bp_channels[bp_name]
            p2p = calculate_bp_peak_to_peak(bp_data, center_idx, fs, config["spike_window_ms"])
            bg_std = np.std(bp_data[train_samples:])
            bp_valid.append(p2p > 1.5 * bg_std)
        
        if all(bp_valid):
            valid_candidates.append(cand)
    
    if len(valid_candidates) == 0:
        print(f"Patient {pid} (test): No valid candidates after FPS filtering, test failed\n")
        return None
    test_candidate_indices = [c["center_idx"] for c in valid_candidates]
    print(f"Patient {pid} (test): Valid candidates after FPS filtering={len(valid_candidates)}")

    feature_matrix = []
    for cand in valid_candidates:
        ch_name = cand["channel"]
        center_idx = cand["center_idx"]
        bp_names = AV_TO_BP_MAP[ch_name]
        
        av_data = cand["segment"]
        if len(av_data) < spike_window_samples:
            pad_left = (spike_window_samples - len(av_data)) // 2
            pad_right = spike_window_samples - len(av_data) - pad_left
            av_data = np.pad(av_data, (pad_left, pad_right), mode='constant')
        elif len(av_data) > spike_window_samples:
            av_data = av_data[:spike_window_samples]
        
        av_features = extract_spike_features(av_data, fs)
        
        bp_features_list = []
        for bp_name in bp_names:
            if bp_name not in bp_channels:
                break
                
            start_idx = max(train_samples, center_idx - spike_window_samples // 2)
            end_idx = min(raw_data.shape[1], center_idx + spike_window_samples // 2)
            bp_data = bp_channels[bp_name][start_idx:end_idx]
            
            if len(bp_data) < spike_window_samples:
                pad_left = (spike_window_samples - len(bp_data)) // 2
                pad_right = spike_window_samples - len(bp_data) - pad_left
                bp_data = np.pad(bp_data, (pad_left, pad_right), mode='constant')
            elif len(bp_data) > spike_window_samples:
                bp_data = bp_data[:spike_window_samples]
                
            bp_features = extract_spike_features(bp_data, fs)
            bp_features_list.append(bp_features)
        
        if len(bp_features_list) != 2:
            continue
        
        full_features = av_features + bp_features_list[0] + bp_features_list[1]
        feature_matrix.append(full_features)
    
    if len(feature_matrix) == 0:
        print(f"Patient {pid} (test): Feature extraction failed, test failed\n")
        return None
    feature_matrix = np.array(feature_matrix)
    print(f"Patient {pid} (test): Successfully extracted 30-dimensional features for {feature_matrix.shape[0]} candidates")

    y_pred = rf_model.predict(feature_matrix)
    matched_candidate_indices = test_candidate_indices[:len(y_pred)]

    TP = 0
    matched_true_spikes = []
    for i, cand_idx in enumerate(matched_candidate_indices):
        if y_pred[i] == 1:
            for true_idx in test_true_spikes:
                if np.abs(cand_idx - true_idx) < spike_window_samples // 2 and true_idx not in matched_true_spikes:
                    TP += 1
                    matched_true_spikes.append(true_idx)
                    break
    
    FP = np.sum(y_pred) - TP
    FN = len(test_true_spikes) - len(matched_true_spikes)
    TN = len(y_pred) - FP
    
    metrics = calculate_paper_metrics(TP, FP, FN, TN)
    print(f"\nPatient {pid} (test): TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"Patient {pid} (test): Sen={metrics['Sen']}%, Spe={metrics['Spe']}%, Sel={metrics['Sel']}%, AC={metrics['AC']}%\n")

    return {
        "pid": pid,
        "metrics": metrics,
        "cm": {"TP": TP, "FP": FP, "FN": FN, "TN": TN}
    }

# === Main Program ===
def main():
    print("="*60)
    print("0. Initial Template vs Real EEG Waveform Segment Comparison Analysis")
    print("="*60)

    print("="*60)
    print("1. Loading and Preprocessing EEG Data")
    print("="*60)
    patient_list = load_and_preprocess_data(DATA_CONFIG)
    if len(patient_list) == 0:
        print("Error: No valid patient data, program exiting")
        return

    print("="*60)
    print("2. Batch Execution of BECT Spike Detection (Training → Testing)")
    print("="*60)
    all_metrics = []
    all_cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    
    for patient in patient_list:
        pid = patient["pid"]
        print(f"{'='*40}\nProcessing patient: {pid}\n{'='*40}")
        
        train_result = train_bect_model(patient, DATA_CONFIG)
        if not train_result:
            continue
        
        test_result = test_bect_model(patient, train_result, DATA_CONFIG)
        if not test_result:
            continue
        
        all_metrics.append(test_result["metrics"])
        for key in all_cm.keys():
            all_cm[key] += test_result["cm"][key]

    print("="*60)
    print("3. Global Detection Results Summary")
    print("="*60)
    if not all_metrics:
        print("Error: No valid test results, cannot summarize")
        return
    
    global_metrics = calculate_paper_metrics(
        all_cm["TP"], all_cm["FP"], all_cm["FN"], all_cm["TN"]
    )
    
    print(f"\nGlobal confusion matrix: TP={all_cm['TP']}, FP={all_cm['FP']}, FN={all_cm['FN']}, TN={all_cm['TN']}")
    print(f"\nGlobal core metrics:")
    print(f"Sensitivity (Sen): {global_metrics['Sen']}%")
    print(f"Specificity (Spe): {global_metrics['Spe']}%")
    print(f"Selectivity (Sel): {global_metrics['Sel']}%")
    print(f"Accuracy (AC): {global_metrics['AC']}%")
    
    print(f"\nIndividual patient test metrics:")
    print(f"{'Patient ID':<12} {'Sen(%)':<8} {'Spe(%)':<8} {'Sel(%)':<8} {'AC(%)':<8}")
    print("-"*50)
    for i, (patient, metrics) in enumerate(zip(patient_list, all_metrics)):
        print(f"{patient['pid']:<12} {metrics['Sen']:<8} {metrics['Spe']:<8} {metrics['Sel']:<8} {metrics['AC']:<8}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()