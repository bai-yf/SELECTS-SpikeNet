# visualization/brain_topography.py
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.text import Text

def plot_brain_topography(raw_data, channel_names, target_channels, channel_counts, sfreq=500):
    values = []
    ch_names_filtered = []
    for ch in target_channels:
        if ch in channel_names and ch in channel_counts:
            values.append(channel_counts[ch]['true_ratio'])
            ch_names_filtered.append(ch)
    
    if not ch_names_filtered:
        return plt.figure()
    
    montage = mne.channels.make_standard_montage('standard_1020')
    pos_dict = montage.get_positions()['ch_pos']
    valid_chs = [ch for ch in ch_names_filtered if ch in pos_dict]
    if not valid_chs:
        return plt.figure()
    
    pos_array = np.array([pos_dict[ch][:2] for ch in valid_chs])
    values_norm = np.array([values[ch_names_filtered.index(ch)] for ch in valid_chs])
    if np.max(values_norm) > 0:
        values_norm = values_norm / np.max(values_norm)
    else:
        values_norm = np.zeros_like(values_norm)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im, _ = mne.viz.plot_topomap(
        values_norm,
        pos=pos_array,
        axes=ax,
        cmap='jet',
        show=False,
        sensors=True
    )
    
    for artist in ax.get_children():
        if isinstance(artist, Line2D) and artist.get_marker() == 'o':
            artist.set_markersize(10)
            artist.set_markeredgewidth(2)
    
    ax.set_title('Brain Topography\n(True Annotation Retention Rate)', fontsize=16, pad=-10)
    cbar = plt.colorbar(im, ax=ax, label='Normalized Rate', shrink=0.9)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    return fig

def plot_electrode_montage(raw_data, channel_names, target_channels, channel_counts, sfreq=500):
    ch_names_filtered = [ch for ch in target_channels if ch in channel_names and ch in channel_counts]
    if not ch_names_filtered:
        return plt.figure()
    
    montage = mne.channels.make_standard_montage('standard_1020')
    pos_dict = montage.get_positions()['ch_pos']
    valid_chs = [ch for ch in ch_names_filtered if ch in pos_dict]
    if not valid_chs:
        return plt.figure()
    
    all_pos = np.array([pos_dict[ch] for ch in valid_chs])
    max_coord = np.max(np.abs(all_pos[:, :2]))
    scale = 1
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    info = mne.create_info(ch_names=valid_chs, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)
    mne.viz.plot_sensors(
        info,
        axes=ax,
        show_names=True,
        title='Electrode Montage (10-20 System)',
        sphere=max_coord * scale
    )
    
    ax.set_xlim(-max_coord * scale, max_coord * scale)
    ax.set_ylim(-max_coord * scale, max_coord * scale)
    ax.set_aspect('equal')
    
    for artist in ax.get_children():
        if isinstance(artist, Line2D) and artist.get_marker() == 'o':
            artist.set_markersize(10)
            artist.set_markeredgewidth(2)
        if isinstance(artist, Text) and artist.get_text() in valid_chs:
            artist.set_fontsize(12)
            artist.set_fontweight('bold')
    
    ax.set_title('Electrode Montage (10-20 System)', fontsize=16, pad=40)
    plt.tight_layout()
    return fig