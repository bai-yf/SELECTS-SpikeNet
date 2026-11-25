
"""Shared configuration and repository-relative paths for the interface."""
import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
INTERFACE_DATA_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", "data"))

CHANNEL_NAMES = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
]
SAMPLING_RATE = 500

MODEL_WEIGHT_PATH = os.path.join(CODE_DIR, "fold_3_best_weights.pth")
EFFICIENT_KAN_PATH = os.path.join(CODE_DIR, "efficient-kan", "src")

EEGLAB_PATH = r"D:\MATLAB\R2024a\toolbox\eeglab"
PROCESSED_DATA_DIR = os.path.join(INTERFACE_DATA_DIR, "set")

MANUAL_SLEEP_STAGE_GUI = os.path.join(CODE_DIR, "manual_sleep_stage_label_gui.py")
MANUAL_SLEEP_LABEL_DIR = os.path.join(CODE_DIR, "manual_sleep_stage_labels")

WINDOW_DURATION = 15
DETECTION_WINDOW_SEC = 0.2
HIGH_PROB_THRESHOLD = 0.8
LOW_PROB_THRESHOLD = 0.5
HIGHLIGHT_CHANNELS = {4, 5, 12, 13}
TARGET_CHANNELS_FOR_TOPOPLOT = ["T3", "T4", "C3", "C4", "P3", "P4", "T5", "T6"]
