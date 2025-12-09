# core/metrics.py
import numpy as np

def calculate_swi(spike_times, total_duration):
    # Fix: Flatten spike_times first (handle possible nested lists), then filter invalid values
    flat_spike_times = []
    for t in spike_times:
        if isinstance(t, list):
            flat_spike_times.extend([x for x in t if isinstance(x, (int, float))])
        elif isinstance(t, (int, float)):
            flat_spike_times.append(t)
    
    valid_spike_times = [t for t in flat_spike_times if 0 <= t < total_duration]
    
    if not valid_spike_times or total_duration <= 0:
        return 0.0, 0, 0
    
    total_seconds = int(np.ceil(total_duration))
    spike_seconds = len(set([int(t) for t in valid_spike_times]))
    
    swi = (spike_seconds / total_seconds) * 100 if total_seconds > 0 else 0
    return round(swi, 2), spike_seconds, total_seconds

def calculate_high_pred1_ratios(raw_data, window_center_times, window_pred_map, 
                               channel_names, target_channels, std_coeff=0.5):
    channel_ratios = {}
    channel_stats = {}
    
    for ch_idx, ch_name in enumerate(channel_names):
        if ch_name not in target_channels:
            continue
        
        win_max_list = []
        win_time_list = []
        for win_time in window_center_times:
            total_time = raw_data.shape[1] / 500
            win_start_idx = int(round(win_time * 500)) - int(0.1 * 500)
            win_end_idx = win_start_idx + int(0.2 * 500)
            win_start_idx = max(0, win_start_idx)
            win_end_idx = min(raw_data.shape[1], win_end_idx)
            
            win_max = np.max(raw_data[ch_idx, win_start_idx:win_end_idx])
            win_max_list.append(win_max)
            win_time_list.append(win_time)
        
        total_windows = len(win_max_list)
        if total_windows == 0:
            channel_ratios[ch_name] = 0.0
            channel_stats[ch_name] = {
                'total_windows': 0,
                'high_windows': 0,
                'high_pred1_windows': 0,
                'ratio': 0.0
            }
            continue
        
        max_mean = np.mean(win_max_list)
        max_std = np.std(win_max_list)
        high_threshold = max_mean + std_coeff * max_std
        
        high_windows = 0
        high_pred1_windows = 0
        for win_time, win_max in zip(win_time_list, win_max_list):
            is_high = (win_max >= high_threshold)
            if not is_high:
                continue
            high_windows += 1
            
            if window_pred_map.get(win_time, 0) == 1:
                high_pred1_windows += 1
        
        ratio = high_pred1_windows / total_windows if total_windows > 0 else 0.0
        channel_ratios[ch_name] = ratio
        channel_stats[ch_name] = {
            'total_windows': total_windows,
            'high_windows': high_windows,
            'high_pred1_windows': high_pred1_windows,
            'ratio': ratio
        }
        
        print(f"Channel {ch_name} Statistics:")
        print(f"  Total windows: {total_windows}")
        print(f"  High-value windows: {high_windows}")
        print(f"  High-value windows with prediction=1: {high_pred1_windows}")
        print(f"  Ratio: {ratio:.2%}\n")
    
    return channel_ratios, channel_stats