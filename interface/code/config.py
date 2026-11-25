# config.py
import os

# EEG channel configuration
CHANNEL_NAMES = ['FP1','FP2','F3','F4','C3','C4','P3','P4',
                 'O1','O2','F7','F8','T3','T4','T5','T6']

# Sampling rate
SAMPLING_RATE = 500  # Hz

# Model paths
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
INTERFACE_DATA_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", "data"))
MODEL_WEIGHT_PATH = os.path.join(CODE_DIR, "fold_3_best_weights.pth")
EFFICIENT_KAN_PATH = os.path.join(CODE_DIR, "efficient-kan", "src")

# MATLAB paths
EEGLAB_PATH = r"D:\MATLAB\R2024a\toolbox\eeglab"
PROCESSED_DATA_DIR = os.path.join(INTERFACE_DATA_DIR, "set")

# Detection parameters
WINDOW_DURATION = 15  # seconds for display
DETECTION_WINDOW_SEC = 0.2  # seconds for detection
HIGH_PROB_THRESHOLD = 0.8
LOW_PROB_THRESHOLD = 0.5

# Visualization
HIGHLIGHT_CHANNELS = {4, 5, 12, 13}  # C3, C4, T3, T4
TARGET_CHANNELS_FOR_TOPOPLOT = ['T3', 'T4', 'C3', 'C4', 'P3', 'P4', 'T5', 'T6']