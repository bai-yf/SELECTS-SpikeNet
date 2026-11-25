
# -*- coding: utf-8 -*-
"""Main window and workflow orchestration for the EEG spike-detection interface."""
import concurrent.futures
import json
import os
import re
import sys
import time

import mne
import numpy as np
import pandas as pd
import pyqtgraph as pg
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph import LinearRegionItem, PlotWidget

from config import CODE_DIR as SCRIPT_DIR, INTERFACE_DATA_DIR, MODEL_WEIGHT_PATH
from core.features import extract_multi_channel_features
from core.metrics import calculate_high_pred1_ratios, calculate_swi
from core.models import FusionModel
from core.preprocessing import (
    call_matlab_preprocess,
    matlab_engine_is_ready,
    shutdown_matlab_engine,
    warm_up_matlab_engine_async,
)
from ui.dialogs import (
    SleepStageImportDialog,
    SpikeConfirmDialog,
    TimeRowRangeDialog,
    read_txt_intervals_fast,
)
from utils.eeg_io import load_eeg_from_source, normalize_eeg_float32, read_supported_eeg
from utils.file_io import load_spike_data, save_spike_data
from visualization.brain_topography import plot_brain_with_electrodes

REVIEW_DIR = os.path.dirname(SCRIPT_DIR)
LONG_TERM_RECORD_CSV = os.path.join(REVIEW_DIR, "长程数据标注记录.csv")
AUTO_RESULT_DIR = os.path.join(SCRIPT_DIR, "results")
MANUAL_CORRECTION_RESULT_DIR = os.path.join(SCRIPT_DIR, "manual_correction_results")
AUTOSAVE_LABEL_DIR = os.path.join(SCRIPT_DIR, "autosave_spike_labels")

for output_dir in (AUTO_RESULT_DIR, MANUAL_CORRECTION_RESULT_DIR, AUTOSAVE_LABEL_DIR):
    os.makedirs(output_dir, exist_ok=True)

LONG_TERM_RECORD_COLUMNS = [
    "编号", "对应原始数据", "总记录时长", "可分析时长", "被排除时长",
    "棘波检测用时", "棘波检测结果", "手动修改核对用时", "手动修改棘波检测结果",
    "人工标注棘波用时", "人工标注棘波结果",
]

INFERENCE_BATCH_SIZE = 256
CPU_INFERENCE_BATCH_SIZE = 64

class EEGAutoSaveProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Spike Detection Tool (TXT/SET/Sleep-Stage Support)")
        self.setGeometry(100, 100, 1650, 1000)

        pg.setConfigOption('foreground', 'd')
        pg.setConfigOption('background', 'w')

        self.raw = None
        self.current_data = None
        # Metadata used by lightweight spike-result files.  The EEG itself stays
        # in its original EDF/SET source instead of being duplicated in the NPZ.
        self.eeg_source_path = None
        self.eeg_source_intervals = []
        self.eeg_source_invert = False
        self.txt_raw_data = None
        self.process_step = 0
        self.channel_names = ['FP1','FP2','F3','F4','C3','C4','P3','P4',
                            'O1','O2','F7','F8','T3','T4','T5','T6']
        self.fs = 500

        self.window_duration = 15
        self.total_samples = 0
        self.samples_per_window = int(self.window_duration * self.fs)

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_weight_path = MODEL_WEIGHT_PATH
        self.window_sec = 0.2
        self.half_window = int(self.window_sec * self.fs / 2)
        self.input_window_size = 2 * self.half_window

        # 新增：窗口中心时间和预测结果映射
        self.window_center_times = []
        self.window_pred_map = {}  # 窗口时间 -> 预测标签(0/1)

        self.artefact_threshold = 11.0
        self.artefact_idx = []

        self.original_y_unit_scale = 1.0
        self.current_y_scale = 1.0
        self.amplitude_scale = 1.0

        self.high_prob_spikes = []
        self.low_prob_spikes = []
        self.current_annotation_nav_index = None
        self.manual_label_mode = False
        self.pending_manual_spike_time = None
        self.selected_spike_times = set()
        self.region_spike_map = {}
        self.file_path = None
        self.last_loaded_spike_path = None
        self.subject_id = None
        self.original_total_duration_sec = None
        self.analyzable_duration_sec = None
        self.excluded_duration_sec = None
        self.last_detection_elapsed_sec = None
        self.manual_correction_elapsed_sec = 0.0
        self.manual_correction_started_at = None
        self.manual_correction_running = False
        self.plot_legend = None

        self.swi_value = 0.0
        self.spike_seconds = 0
        self.total_seconds = 0

        self.init_ui()
        self.init_model()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 左侧功能按钮面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_panel.setStyleSheet("padding: 20px;")
        left_panel.setFixedWidth(380)

        self.import_sleep_btn = QPushButton("0. Import by Sleep Stage")
        self.import_sleep_btn.setFixedSize(340, 60)
        self.import_sleep_btn.setStyleSheet("font-size: 24px; padding: 12px; background-color: #FF5722; color: white;")
        self.import_sleep_btn.clicked.connect(self.import_data_by_sleep_stage)
        left_layout.addWidget(self.import_sleep_btn)

        self.import_raw_btn = QPushButton("1. Import Raw Data")
        self.import_raw_btn.setFixedSize(340, 60)
        self.import_raw_btn.setStyleSheet("font-size: 24px; padding: 12px; background-color: #2196F3; color: white;")
        self.import_raw_btn.clicked.connect(self.import_raw_data)
        left_layout.addWidget(self.import_raw_btn)

        self.import_processed_btn = QPushButton("2. Import Preprocessed")
        self.import_processed_btn.setFixedSize(340, 60)
        self.import_processed_btn.setStyleSheet("font-size: 24px; padding: 12px; background-color: #9C27B0; color: white;")
        self.import_processed_btn.clicked.connect(self.import_processed_data)
        left_layout.addWidget(self.import_processed_btn)

        self.preprocess_btn = QPushButton("3. Run Preprocessing")
        self.preprocess_btn.setFixedSize(340, 60)
        self.preprocess_btn.setStyleSheet("font-size: 24px; padding: 12px; background-color: #7B3294; color: white;")
        self.preprocess_btn.clicked.connect(self.run_preprocessing)
        self.preprocess_btn.setEnabled(False)
        left_layout.addWidget(self.preprocess_btn)

        self.spike_detect_btn = QPushButton("4. Detect Spikes")
        self.spike_detect_btn.setFixedSize(340, 60)
        self.spike_detect_btn.setStyleSheet("font-size: 24px; padding: 12px; background-color: #FF9800; color: white;")
        self.spike_detect_btn.clicked.connect(self.run_spike_detection)
        self.spike_detect_btn.setEnabled(False)
        left_layout.addWidget(self.spike_detect_btn)

        self.recalc_swi_btn = QPushButton("5. Recalculate SWI")
        self.recalc_swi_btn.setFixedSize(340, 60)
        self.recalc_swi_btn.setStyleSheet("font-size: 24px; padding: 12px; background-color: #E69F00; color: white;")
        self.recalc_swi_btn.clicked.connect(self.recalculate_swi)
        self.recalc_swi_btn.setEnabled(False)
        left_layout.addWidget(self.recalc_swi_btn)

        # 脑地形图按钮
        self.plot_brain_btn = QPushButton("6. View Spatial Distribution")
        self.plot_brain_btn.setFixedSize(340, 60)
        self.plot_brain_btn.setStyleSheet("font-size: 22px; padding: 12px; background-color: #7B3294; color: white;")
        self.plot_brain_btn.clicked.connect(self.show_brain_topography)
        self.plot_brain_btn.setEnabled(False)
        left_layout.addWidget(self.plot_brain_btn)

        self.save_spike_btn = QPushButton("7. Save Spike Results")
        self.save_spike_btn.setFixedSize(340, 60)
        self.save_spike_btn.setStyleSheet("font-size: 24px; padding: 12px; background-color: #2196F3; color: white;")
        self.save_spike_btn.clicked.connect(self.save_spike_click)
        self.save_spike_btn.setEnabled(False)
        left_layout.addWidget(self.save_spike_btn)

        self.load_spike_btn = QPushButton("8. Load Spike Results")
        self.load_spike_btn.setFixedSize(340, 60)
        self.load_spike_btn.setStyleSheet("font-size: 24px; padding: 12px; background-color: #9C27B0; color: white;")
        self.load_spike_btn.clicked.connect(self.load_spike_click)
        left_layout.addWidget(self.load_spike_btn)

        self.manual_correction_start_btn = QPushButton("Start Review Timer")
        self.manual_correction_start_btn.setFixedSize(340, 54)
        self.manual_correction_start_btn.setStyleSheet("font-size: 22px; padding: 10px; background-color: #455A64; color: white;")
        self.manual_correction_start_btn.clicked.connect(self.start_manual_correction_timer)
        left_layout.addWidget(self.manual_correction_start_btn)

        self.manual_correction_pause_btn = QPushButton("Pause Review Timer")
        self.manual_correction_pause_btn.setFixedSize(340, 54)
        self.manual_correction_pause_btn.setStyleSheet("font-size: 22px; padding: 10px; background-color: #795548; color: white;")
        self.manual_correction_pause_btn.clicked.connect(self.pause_manual_correction_timer)
        self.manual_correction_pause_btn.setEnabled(False)
        left_layout.addWidget(self.manual_correction_pause_btn)

        self.manual_correction_end_btn = QPushButton("Finish and Save Review")
        self.manual_correction_end_btn.setFixedSize(340, 54)
        self.manual_correction_end_btn.setStyleSheet("font-size: 22px; padding: 10px; background-color: #7B3294; color: white;")
        self.manual_correction_end_btn.clicked.connect(self.end_manual_correction_timer)
        self.manual_correction_end_btn.setEnabled(False)
        left_layout.addWidget(self.manual_correction_end_btn)
        left_layout.addStretch()
        main_layout.addWidget(left_panel, stretch=1)

        # 右侧显示区域
        self.right_panel = QScrollArea()
        self.right_content = QWidget()
        self.right_layout = QVBoxLayout(self.right_content)

        self.plot_widget = PlotWidget(title="EEG Waveform View (purple = high-probability spikes, orange = low-probability spikes; click orange regions to confirm/reject)")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMinimumHeight(600)
        self.plot_widget.setFocusPolicy(Qt.StrongFocus)
        self.plot_widget.setToolTip("Mouse wheel: move left/right by one 15-second window")
        self.plot_widget.viewport().installEventFilter(self)
        self.plot_widget.keyPressEvent = self.keyPressEvent
        self._original_plot_mouse_press_event = self.plot_widget.mousePressEvent
        self.plot_widget.mousePressEvent = self.on_plot_mouse_press
        self.plot_widget.scene().sigMouseClicked.connect(self.on_plot_click)
        self.plot_legend = self.plot_widget.addLegend(offset=(10, 10), labelTextSize='16pt')
        self.right_layout.addWidget(self.plot_widget)

        left_axis = self.plot_widget.getAxis('left')
        left_axis.setWidth(58)
        left_axis.setStyle(tickLength=0, tickTextOffset=8)
        bottom_axis = self.plot_widget.getAxis('bottom')
        bottom_axis.setStyle(tickFont=QFont("Arial", 10), tickTextOffset=4)
        self.plot_widget.setLabel('left', '')

        control_style = "font-size: 14px;"
        control_button_style = "font-size: 14px; padding: 2px 6px;"
        amplitude_layout = QHBoxLayout()
        amplitude_label = QLabel("Amplitude Scale:")
        amplitude_label.setStyleSheet(control_style)
        amplitude_layout.addWidget(amplitude_label)
        self.decrease_amp_btn = QPushButton("Amp -")
        self.decrease_amp_btn.setFixedSize(100, 36)
        self.decrease_amp_btn.setStyleSheet(control_button_style)
        self.decrease_amp_btn.clicked.connect(lambda: self.adjust_amplitude_scale(-0.2))
        amplitude_layout.addWidget(self.decrease_amp_btn)
        self.amp_scale_label = QLabel(f"{self.amplitude_scale:.1f}x")
        self.amp_scale_label.setMinimumWidth(55)
        self.amp_scale_label.setStyleSheet(control_style)
        amplitude_layout.addWidget(self.amp_scale_label)
        self.increase_amp_btn = QPushButton("Amp +")
        self.increase_amp_btn.setFixedSize(100, 36)
        self.increase_amp_btn.setStyleSheet(control_button_style)
        self.increase_amp_btn.clicked.connect(lambda: self.adjust_amplitude_scale(0.2))
        amplitude_layout.addWidget(self.increase_amp_btn)
        self.reset_amp_btn = QPushButton("Reset")
        self.reset_amp_btn.setFixedSize(90, 36)
        self.reset_amp_btn.setStyleSheet(control_button_style)
        self.reset_amp_btn.clicked.connect(self.reset_amplitude_scale)
        amplitude_layout.addWidget(self.reset_amp_btn)
        amplitude_layout.addStretch()
        self.right_layout.addLayout(amplitude_layout)

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setEnabled(False)
        self.time_slider.setToolTip("Drag to control the displayed 15-second time window")
        self.time_slider.keyPressEvent = self.keyPressEvent
        self.time_slider.valueChanged.connect(self.update_plot_by_slider)
        time_range_label = QLabel("Time Range Control (15-second window):")
        time_range_label.setStyleSheet(control_style)
        self.right_layout.addWidget(time_range_label)
        self.right_layout.addWidget(self.time_slider)

        self.swi_label = QLabel("Spike-Wave Index (SWI)\nWaiting for detection or loaded results...")
        self.swi_label.setStyleSheet("""
            font-size: 24px; margin: 6px; padding: 20px;
            border: 2px solid #7B3294; border-radius: 8px;
            background-color: #F3EEF8; white-space: pre-wrap;
        """)
        self.right_layout.addWidget(self.swi_label)

        self.status_label = QLabel("Status: waiting for data import | loading model...")
        self.status_label.setStyleSheet("font-size: 24px; margin: 6px; padding: 16px; border: 1px solid #eee; border-radius: 5px;")
        self.right_layout.addWidget(self.status_label)

        self.info_label = QLabel("Tips:\n1. Raw TXT or SET data can be imported.\n2. Import by Sleep Stage can automatically extract N1/N2/N3 segments.\n3. After detection, click orange regions to confirm them as purple or reject them.\n4. Click Recalculate SWI after manual edits.\n5. Click View Spatial Distribution after detection to inspect spatial distribution.")
        self.info_label.setStyleSheet('''font-size: 24px; margin: 6px; padding: 20px;
            border: 2px solid #7B3294; border-radius: 8px;
            background-color: #F3EEF8; white-space: pre-wrap;
            ''')
        self.right_layout.addWidget(self.info_label)

        self.right_panel.setWidget(self.right_content)
        self.right_panel.setWidgetResizable(True)
        main_layout.addWidget(self.right_panel, stretch=4)

    def format_duration(self, seconds):
        if seconds is None or pd.isna(seconds):
            return ""
        seconds = max(0, int(round(float(seconds))))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def load_long_term_record_df(self):
        if os.path.exists(LONG_TERM_RECORD_CSV):
            try:
                df = pd.read_csv(LONG_TERM_RECORD_CSV, dtype=str, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(LONG_TERM_RECORD_CSV, dtype=str, encoding="gbk")
        else:
            df = pd.DataFrame(columns=LONG_TERM_RECORD_COLUMNS)
        for col in LONG_TERM_RECORD_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[LONG_TERM_RECORD_COLUMNS].fillna("")
        for time_col, result_col in [
            ("棘波检测用时", "棘波检测结果"),
            ("手动修改核对用时", "手动修改棘波检测结果"),
        ]:
            path_mask = df[time_col].astype(str).apply(lambda value: ":\\\\" in value or value.startswith("\\\\\\\\"))
            empty_result_mask = df[result_col].astype(str).str.strip().eq("")
            move_mask = path_mask & empty_result_mask
            df.loc[move_mask, result_col] = df.loc[move_mask, time_col]
            df.loc[path_mask, time_col] = ""
        return df

    def save_long_term_record_df(self, df):
        os.makedirs(os.path.dirname(LONG_TERM_RECORD_CSV), exist_ok=True)
        df = df[LONG_TERM_RECORD_COLUMNS].fillna("")
        try:
            df.to_csv(LONG_TERM_RECORD_CSV, index=False, encoding="utf-8-sig")
        except PermissionError:
            QMessageBox.warning(
                self,
                "Record File In Use",
                f"Could not write the long-term annotation record. It may be open in Excel/WPS.\nPlease close it and retry:\n{LONG_TERM_RECORD_CSV}"
            )

    def ensure_subject_id(self):
        if not self.file_path:
            return None
        filename = os.path.basename(self.file_path)
        filename_match = re.match(r"^(S\d{3})(?:[_\-.]|$)", filename, re.IGNORECASE)
        if filename_match:
            self.subject_id = filename_match.group(1).upper()
            return self.subject_id

        df = self.load_long_term_record_df()
        file_path_norm = os.path.normcase(os.path.abspath(self.file_path))
        for _, row in df.iterrows():
            old_path = str(row.get("对应原始数据", ""))
            if old_path and os.path.normcase(os.path.abspath(old_path)) == file_path_norm:
                self.subject_id = str(row.get("编号", "")).strip()
                return self.subject_id

        used = set(str(x).strip() for x in df["编号"].tolist() if str(x).strip())
        idx = 1
        while f"S{idx:03d}" in used:
            idx += 1
        self.subject_id = f"S{idx:03d}"
        return self.subject_id

    def update_long_term_record(self, **updates):
        if not self.file_path:
            return
        subject_id = self.ensure_subject_id()
        if not subject_id:
            return
        df = self.load_long_term_record_df()
        if subject_id in df["编号"].astype(str).tolist():
            row_idx = df.index[df["编号"].astype(str) == subject_id][0]
        else:
            row_idx = len(df)
            df.loc[row_idx, LONG_TERM_RECORD_COLUMNS] = ""
            df.at[row_idx, "编号"] = subject_id

        source_path = os.path.abspath(self.file_path)
        existing_source_path = str(df.at[row_idx, "对应原始数据"]).strip()
        defaults = {
            "总记录时长": self.format_duration(self.original_total_duration_sec),
            "可分析时长": self.format_duration(self.analyzable_duration_sec),
            "被排除时长": self.format_duration(self.excluded_duration_sec),
        }
        if not existing_source_path:
            defaults["对应原始数据"] = source_path
        elif os.path.normcase(os.path.abspath(existing_source_path)) == os.path.normcase(source_path):
            defaults["对应原始数据"] = source_path
        defaults.update(updates)
        for col, value in defaults.items():
            if col in LONG_TERM_RECORD_COLUMNS and value not in (None, ""):
                df.at[row_idx, col] = str(value)
        self.save_long_term_record_df(df)

    def get_result_path(self, root_dir, suffix):
        subject_id = self.ensure_subject_id() or "S000"
        subject_dir = os.path.join(root_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        return os.path.join(subject_dir, f"{subject_id}_{suffix}.npz")

    def get_autosave_subject_key(self):
        subject_id = self.ensure_subject_id()
        if subject_id:
            return subject_id
        if self.last_loaded_spike_path:
            stem = os.path.splitext(os.path.basename(self.last_loaded_spike_path))[0]
            match = re.match(r"(S\d{3})_", stem, re.IGNORECASE)
            return match.group(1).upper() if match else stem
        if self.file_path:
            return re.sub(r"[^0-9A-Za-z_-]+", "_", os.path.splitext(os.path.basename(self.file_path))[0])
        if self.total_samples and self.fs:
            return f"unknown_{int(round(float(self.fs)))}hz_{int(self.total_samples)}samples"
        return None

    def get_autosave_label_path(self):
        key = self.get_autosave_subject_key()
        if not key:
            return None
        safe_key = re.sub(r"[^0-9A-Za-z_-]+", "_", str(key))
        return os.path.join(AUTOSAVE_LABEL_DIR, f"{safe_key}_autosave.json")

    def save_spike_label_autosave(self, reason="manual_edit"):
        if self.current_data is None or not self.total_samples:
            return
        autosave_path = self.get_autosave_label_path()
        if not autosave_path:
            return
        payload = {
            "version": 1,
            "reason": reason,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "subject_id": self.get_autosave_subject_key(),
            "source_file_path": os.path.abspath(self.file_path) if self.file_path else "",
            "loaded_spike_result_path": os.path.abspath(self.last_loaded_spike_path) if self.last_loaded_spike_path else "",
            "fs": float(self.fs),
            "total_samples": int(self.total_samples),
            "high_prob_spikes": [[float(t), float(prob)] for t, prob in self.high_prob_spikes],
            "low_prob_spikes": [[float(t), float(prob)] for t, prob in self.low_prob_spikes],
        }
        try:
            os.makedirs(AUTOSAVE_LABEL_DIR, exist_ok=True)
            tmp_path = autosave_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, autosave_path)
        except Exception as e:
            self.status_label.setText(f"Status: autosave draft failed: {str(e)}")

    def clear_spike_label_autosave(self):
        autosave_path = self.get_autosave_label_path()
        if autosave_path and os.path.exists(autosave_path):
            try:
                os.remove(autosave_path)
            except OSError:
                pass

    def find_matching_spike_label_autosave(self):
        if self.current_data is None or not self.total_samples:
            return None, None
        candidates = []
        direct_path = self.get_autosave_label_path()
        if direct_path:
            candidates.append(direct_path)
        if os.path.isdir(AUTOSAVE_LABEL_DIR):
            for name in os.listdir(AUTOSAVE_LABEL_DIR):
                if name.endswith("_autosave.json"):
                    path = os.path.join(AUTOSAVE_LABEL_DIR, name)
                    if path not in candidates:
                        candidates.append(path)

        current_source = os.path.normcase(os.path.abspath(self.file_path)) if self.file_path else ""
        current_subject = self.get_autosave_subject_key()
        matches = []
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if int(payload.get("total_samples", -1)) != int(self.total_samples):
                    continue
                if abs(float(payload.get("fs", -1)) - float(self.fs)) > 1e-6:
                    continue
                saved_subject = str(payload.get("subject_id", "")).strip()
                if current_subject and saved_subject and saved_subject != current_subject:
                    continue
                saved_source = str(payload.get("source_file_path", "")).strip()
                if current_source and saved_source and os.path.normcase(os.path.abspath(saved_source)) != current_source:
                    continue
                matches.append((os.path.getmtime(path), path, payload))
            except Exception:
                continue
        if not matches:
            return None, None
        _, path, payload = max(matches, key=lambda x: x[0])
        return path, payload

    def maybe_restore_spike_label_autosave(self):
        autosave_path, payload = self.find_matching_spike_label_autosave()
        if not payload:
            return False
        high_count = len(payload.get("high_prob_spikes", []))
        low_count = len(payload.get("low_prob_spikes", []))
        saved_at = payload.get("saved_at", "unknown time")
        reply = QMessageBox.question(
            self,
            "Unsaved Draft Found",
            f"Found an unsaved label draft from the previous session:\nSaved at: {saved_at}\nPurple labels: {high_count}, orange labels: {low_count}\n\nRestore this draft?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return False

        self.high_prob_spikes = [(float(t), float(prob)) for t, prob in payload.get("high_prob_spikes", [])]
        self.low_prob_spikes = [(float(t), float(prob)) for t, prob in payload.get("low_prob_spikes", [])]
        self.high_prob_spikes.sort(key=lambda x: x[0])
        self.low_prob_spikes.sort(key=lambda x: x[0])
        self.selected_spike_times.clear()
        self.current_annotation_nav_index = None
        if self.window_pred_map:
            for key in list(self.window_pred_map.keys()):
                self.window_pred_map[key] = 0
            for spike_time, _ in self.high_prob_spikes + self.low_prob_spikes:
                nearest_time = min(self.window_center_times, key=lambda t: abs(t - spike_time)) if self.window_center_times else None
                if nearest_time is not None:
                    self.window_pred_map[nearest_time] = 1
        self.process_step = 3
        self.save_spike_btn.setEnabled(True)
        self.recalc_swi_btn.setEnabled(True)
        self.plot_brain_btn.setEnabled(True)
        self.recalculate_swi()
        self.update_plot_by_slider(self.time_slider.value())
        self.status_label.setText(
            f"Status: restored unsaved draft | purple {len(self.high_prob_spikes)}, orange {len(self.low_prob_spikes)}"
        )
        return True

    def save_current_spikes_to_path(self, save_path, show_dialog=True):
        all_spike_times = [t for t, _ in self.high_prob_spikes] + [t for t, _ in self.low_prob_spikes]
        if self.current_data is None:
            QMessageBox.warning(self, "Save Failed", "No EEG data available to save.")
            return None
        return save_spike_data(
            all_spike_times,
            self.current_data,
            self.fs,
            save_path=save_path,
            show_dialog=show_dialog,
            parent=self,
            high_prob_spikes=self.high_prob_spikes,
            low_prob_spikes=self.low_prob_spikes,
            total_record_duration=getattr(self, "original_total_duration_sec", None),
            analyzable_duration=getattr(self, "analyzable_duration_sec", None),
            excluded_duration=getattr(self, "excluded_duration_sec", None),
            spike_detection_elapsed=getattr(self, "last_detection_elapsed_sec", None),
            source_file_path=getattr(self, "eeg_source_path", None),
            source_intervals=getattr(self, "eeg_source_intervals", []),
            source_invert=getattr(self, "eeg_source_invert", False),
            source_channel_names=self.channel_names[:self.current_data.shape[0]]
        )

    def start_manual_correction_timer(self):
        if self.current_data is None:
            QMessageBox.warning(self, "Cannot Start", "Please import or load EEG data first.")
            return
        if self.manual_correction_running:
            return
        self.ensure_subject_id()
        self.manual_correction_started_at = time.time()
        self.manual_correction_running = True
        self.manual_correction_start_btn.setEnabled(False)
        self.manual_correction_pause_btn.setEnabled(True)
        self.manual_correction_end_btn.setEnabled(True)
        self.status_label.setText("Status: manual review timer running")

    def pause_manual_correction_timer(self):
        if not self.manual_correction_running:
            return
        self.manual_correction_elapsed_sec += time.time() - self.manual_correction_started_at
        self.manual_correction_started_at = None
        self.manual_correction_running = False
        self.manual_correction_start_btn.setText("Resume Review Timer")
        self.manual_correction_start_btn.setEnabled(True)
        self.manual_correction_pause_btn.setEnabled(False)
        self.manual_correction_end_btn.setEnabled(True)
        elapsed = self.format_duration(self.manual_correction_elapsed_sec)
        self.status_label.setText(f"Status: manual review timer paused | elapsed {elapsed}")

    def end_manual_correction_timer(self):
        if self.manual_correction_running:
            self.manual_correction_elapsed_sec += time.time() - self.manual_correction_started_at
            self.manual_correction_started_at = None
            self.manual_correction_running = False
        elapsed = self.format_duration(self.manual_correction_elapsed_sec)
        save_path = self.get_result_path(MANUAL_CORRECTION_RESULT_DIR, "manual_correction")
        saved_path = self.save_current_spikes_to_path(save_path, show_dialog=False)
        if not saved_path:
            return
        self.update_long_term_record(
            **{
                "手动修改核对用时": elapsed,
                "手动修改棘波检测结果": saved_path,
            }
        )
        self.manual_correction_start_btn.setText("Start Review Timer")
        self.manual_correction_start_btn.setEnabled(True)
        self.manual_correction_pause_btn.setEnabled(False)
        self.manual_correction_end_btn.setEnabled(False)
        QMessageBox.information(
            self,
            "Manual Review Saved",
            f"Manual review elapsed time: {elapsed}\nResults saved to:\n{saved_path}\n\nRecord updated:\n{LONG_TERM_RECORD_CSV}"
        )
        self.status_label.setText(f"Status: manual review complete | elapsed {elapsed}")
        self.manual_correction_elapsed_sec = 0.0
        self.manual_correction_started_at = None
        self.manual_correction_running = False
        self.clear_spike_label_autosave()
    # 显示脑地形图和电极定位图（核心修改）
    def show_brain_topography(self):
        """显示电极定位叠加在脑地形图上的合并图"""
        if self.process_step != 3 or not self.window_center_times or not self.window_pred_map:
            QMessageBox.warning(self, "Requirements Not Met", "Please finish spike detection first and make sure window predictions are available.")
            return

        target_channels = ['T3', 'T4', 'C3', 'C4', 'P3', 'P4', 'T5', 'T6']

        # 计算通道比例
        try:
            channel_ratios, _ = calculate_high_pred1_ratios(
                raw_data=self.current_data,
                window_center_times=self.window_center_times,
                window_pred_map=self.window_pred_map,
                channel_names=self.channel_names,
                target_channels=target_channels,
                std_coeff=0.5
            )
        except Exception as e:
            QMessageBox.critical(self, "Statistics Failed", f"Error while calculating ratios: {str(e)}")
            return

        channel_counts = {ch: {'true_ratio': ratio} for ch, ratio in channel_ratios.items()}

        # 绘制“定位叠在地形图上”的合并图
        combined_fig = plot_brain_with_electrodes(
            raw_data=self.current_data,
            channel_names=self.channel_names,
            target_channels=target_channels,
            channel_counts=channel_counts,
            sfreq=self.fs
        )

        # 弹窗显示结果
        dialog = QDialog(self)
        dialog.setWindowTitle("Channel-Level Spatial Distribution")
        dialog.setGeometry(200, 100, 800, 800)  # 正方形窗口适配圆形脑图
        layout = QVBoxLayout(dialog)
        canvas = FigureCanvas(combined_fig)
        layout.addWidget(canvas)
        btn = QDialogButtonBox(QDialogButtonBox.Close)
        close_button = btn.button(QDialogButtonBox.Close)
        if close_button is not None:
            close_button.setStyleSheet("font-size: 12px; padding: 4px 10px;")
        btn.rejected.connect(dialog.close)
        layout.addWidget(btn)
        dialog.show()

    def import_data_by_sleep_stage(self):
        sleep_dialog = SleepStageImportDialog(self)
        if sleep_dialog.exec_() != QDialog.Accepted:
            return

        data_file_path = getattr(sleep_dialog, 'data_file_path', sleep_dialog.txt_file_path)
        start_row = sleep_dialog.selected_start_row
        end_row = sleep_dialog.selected_end_row
        start_time = getattr(sleep_dialog, 'selected_start_time', None)
        end_time = getattr(sleep_dialog, 'selected_end_time', None)
        total_rows = sleep_dialog.total_rows
        sleep_intervals = getattr(sleep_dialog, 'sleep_intervals', [])
        self.fs = sleep_dialog.fs
        self.samples_per_window = int(self.window_duration * self.fs)
        file_ext = os.path.splitext(data_file_path)[1].lower()

        try:
            prepared_current_data = None
            prepared_invert = False
            prepared_is_normalized = False
            messages = []
            over_16_count = 0
            invalid_count = 0
            over_16_samples = []
            invalid_samples = []

            if file_ext == ".txt":
                interval_rows = []
                for seg_start, seg_end, seg_stage in sleep_intervals:
                    row_start = max(1, int(((sleep_dialog.base_offset or 0) + seg_start) * self.fs) + 1)
                    row_end = max(row_start, int(((sleep_dialog.base_offset or 0) + seg_end) * self.fs))
                    interval_rows.append((row_start, row_end, seg_stage))
                txt_size_gb = os.path.getsize(data_file_path) / (1024 ** 3)
                import_progress = QProgressDialog(
                    f"Reading {os.path.basename(data_file_path)} ({txt_size_gb:.2f} GB)...\n"
                    "The first import normally takes about 15-30 seconds.",
                    None,
                    0,
                    0,
                    self,
                )
                import_progress.setWindowTitle("Importing TXT Data")
                import_progress.setWindowModality(Qt.WindowModal)
                import_progress.setMinimumDuration(0)
                import_progress.setAutoClose(False)
                import_progress.show()
                self.status_label.setText(
                    f"Status: reading {os.path.basename(data_file_path)} in background | model loaded"
                )
                QApplication.processEvents()
                try:
                    # Keep the Qt event loop responsive while the large, one-time
                    # disk scan and the parallel pandas parsers run off the GUI thread.
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as import_executor:
                        import_future = import_executor.submit(
                            read_txt_intervals_fast,
                            data_file_path,
                            interval_rows,
                            16,
                        )
                        while not import_future.done():
                            QApplication.processEvents()
                            time.sleep(0.03)
                        data_np, invalid_count, header_rows = import_future.result()
                finally:
                    import_progress.close()
                    QApplication.processEvents()
                if data_np.shape[1] == 0:
                    raise ValueError("No valid TXT data were found within the N1/N2/N3 segment range.")
                if data_np.shape[0] != 16:
                    raise ValueError(f"Invalid channel count: expected 16, got {data_np.shape[0]}.")

                # Keep the one-time TXT extraction in memory only.  A separate
                # copy is normalized for display; the raw float32 data is kept
                # until Run Preprocessing exports its temporary SET file.
                self.txt_raw_data = data_np
                self.raw = None
                prepared_current_data = np.array(data_np, dtype=np.float32, copy=True)
                prepared_invert = False
                prepared_is_normalized = False
                range_text = (
                    f"Merged N1/N2/N3 segments: {len(sleep_intervals)} segments, "
                    f"{data_np.shape[1]} rows (fast indexed TXT reader, header rows: {header_rows})"
                )
                next_step_text = "Next step: click Run Preprocessing"
                self.process_step = 1
                self.preprocess_btn.setEnabled(True)
                self.spike_detect_btn.setEnabled(False)
            elif file_ext in (".edf", ".set", ".fdt"):
                data_file_path = resolve_eeg_file_path(data_file_path)
                file_ext = os.path.splitext(data_file_path)[1].lower()
                # Reuse the header object opened in the selection dialog.  Large
                # EEGLAB SET headers can otherwise be parsed twice.
                raw_full = getattr(sleep_dialog, "raw_header", None)
                if raw_full is None:
                    raw_full = read_supported_eeg(data_file_path, preload=False, verbose=False)

                def norm_ch(name):
                    text = re.sub(r'[^A-Z0-9]', '', str(name).upper())
                    for token in ('EEG', 'REF', 'LE', 'AVG'):
                        text = text.replace(token, '')
                    return text

                available = {norm_ch(ch): ch for ch in raw_full.ch_names}
                picks = []
                missing = []
                for wanted in self.channel_names:
                    wanted_norm = norm_ch(wanted)
                    if wanted_norm in available:
                        picks.append(available[wanted_norm])
                    else:
                        match = next((ch for ch in raw_full.ch_names if wanted_norm in norm_ch(ch)), None)
                        if match:
                            picks.append(match)
                        else:
                            missing.append(wanted)
                if missing:
                    raise ValueError(f"Missing channels in EDF: {', '.join(missing)}")

                raw_full.pick_channels(picks, ordered=True)
                rename_map = {old: new for old, new in zip(picks, self.channel_names) if old != new}
                if rename_map:
                    raw_full.rename_channels(rename_map)

                raw_fs = float(raw_full.info['sfreq'])
                total_duration = raw_full.n_times / raw_fs
                segment_specs = []
                kept_intervals = []
                for seg_start, seg_end, seg_stage in sleep_intervals:
                    tmin = max(0.0, float(seg_start))
                    tmax = min(total_duration, float(seg_end))
                    if tmax <= tmin:
                        continue
                    start_sample = max(0, int(round(tmin * raw_fs)))
                    end_sample = min(raw_full.n_times, int(round(tmax * raw_fs)))
                    if end_sample <= start_sample:
                        continue
                    segment_specs.append((start_sample, end_sample))
                    kept_intervals.append((tmin, tmax, seg_stage))
                if not segment_specs:
                    raise ValueError("No valid EDF data were found within the N1/N2/N3 segment range.")

                # Allocate the final float32 array once and stream directly into
                # it.  This avoids retaining all float64 segments, concatenating
                # them, wrapping them in RawArray, and copying them out again.
                total_kept_samples = sum(end - start for start, end in segment_specs)
                data_np = np.empty((len(picks), total_kept_samples), dtype=np.float32)
                # Ten-minute chunks are a good balance for long EEGLAB FDT files:
                # roughly 38 MB temporary float64 data for 16 channels at 500 Hz.
                chunk_samples = max(1, int(raw_fs * 600.0))
                write_pos = 0
                for start_sample, end_sample in segment_specs:
                    for chunk_start in range(start_sample, end_sample, chunk_samples):
                        chunk_end = min(end_sample, chunk_start + chunk_samples)
                        block = raw_full.get_data(start=chunk_start, stop=chunk_end)
                        block_len = chunk_end - chunk_start
                        data_np[:, write_pos:write_pos + block_len] = block
                        write_pos += block_len
                prepared_current_data = data_np
                prepared_invert = True
                prepared_is_normalized = False
                self.txt_raw_data = None
                self.raw = None
                self.fs = raw_fs
                self.samples_per_window = int(self.window_duration * self.fs)
                total_kept_sec = data_np.shape[1] / self.fs
                range_text = f"Merged N1/N2/N3 segments: {len(kept_intervals)} segments, {total_kept_sec:.2f} s"
                next_step_text = "Next step: click Detect Spikes"
                self.process_step = 2
                self.preprocess_btn.setEnabled(False)
                self.spike_detect_btn.setEnabled(True if self.model is not None else False)
            else:
                raise ValueError(f"Unsupported format: {file_ext} (TXT/EDF/SET only)")

            self.file_path = data_file_path
            self.eeg_source_path = data_file_path
            self.eeg_source_intervals = (
                [(float(start), float(end)) for start, end, _ in kept_intervals]
                if file_ext in (".edf", ".set") else []
            )
            self.eeg_source_invert = file_ext in (".edf", ".set")
            self.last_loaded_spike_path = None
            if prepared_current_data is not None:
                if prepared_is_normalized:
                    self.current_data = prepared_current_data
                else:
                    self.current_data = normalize_eeg_float32(
                        prepared_current_data,
                        invert=prepared_invert,
                    )
            else:
                self.current_data = normalize_eeg_float32(self.raw.get_data(), invert=False)

            self.total_samples = self.current_data.shape[1]
            self.analyzable_duration_sec = self.total_samples / self.fs if self.fs else None
            self.original_total_duration_sec = total_rows / self.fs if total_rows and self.fs else self.analyzable_duration_sec
            self.excluded_duration_sec = (
                max(0.0, self.original_total_duration_sec - self.analyzable_duration_sec)
                if self.original_total_duration_sec is not None and self.analyzable_duration_sec is not None
                else None
            )
            self.ensure_subject_id()
            self.update_long_term_record()
            self.last_detection_elapsed_sec = None
            self.manual_correction_elapsed_sec = 0.0
            self.manual_correction_started_at = None
            self.manual_correction_running = False
            self.manual_correction_start_btn.setText("Start Review Timer")
            self.manual_correction_start_btn.setEnabled(True)
            self.manual_correction_pause_btn.setEnabled(False)
            self.manual_correction_end_btn.setEnabled(False)
            max_slider_value = max(0, self.total_samples - self.samples_per_window)
            self.time_slider.setMaximum(max_slider_value)
            self.time_slider.setEnabled(True)

            self.high_prob_spikes.clear()
            self.low_prob_spikes.clear()
            self.current_annotation_nav_index = None
            self.region_spike_map.clear()
            self.window_center_times.clear()
            self.window_pred_map.clear()
            self.swi_value = 0.0
            self.spike_seconds = 0
            self.total_seconds = 0
            self.swi_label.setText("Spike-Wave Index (SWI)\nWaiting for detection or loaded results...")
            self.save_spike_btn.setEnabled(False)
            self.recalc_swi_btn.setEnabled(False)
            self.plot_brain_btn.setEnabled(False)

            self.update_plot_by_slider(0)
            file_type = "SET" if file_ext == ".set" else ("EDF" if file_ext == ".edf" else "TXT")
            self.status_label.setText(f"Status: imported {file_type} data by sleep stage | model loaded")
            self.info_label.setText(
                f"Import by sleep stage succeeded.\n"
                f"File: {os.path.basename(data_file_path)}\n"
                f"{range_text}\n"
                f"Channels: 16 | Sampling rate: {self.fs:g} Hz\n"
                f"Data duration: {self.total_samples/self.fs:.2f} s\n"
                f"{next_step_text}"
            )

            if over_16_count > 0:
                rows_str = ", ".join(map(str, over_16_samples[:5]))
                if over_16_count > 5:
                    rows_str += f" and {over_16_count} rows total"
                messages.append(f"Detected {over_16_count} rows with more than 16 channels: {rows_str}")
                messages.append("Only the first 16 channels were kept for analysis.")

            if invalid_count > 0:
                if invalid_samples:
                    rows_str = ", ".join(map(str, invalid_samples[:5]))
                    if invalid_count > 5:
                        rows_str += f" and {invalid_count} rows total"
                else:
                    rows_str = "row numbers were not retained by the fast TXT reader"
                messages.append(f"Found {invalid_count} rows with fewer than 16 channels: {rows_str}")
                messages.append("These rows were skipped and excluded from analysis.")

            if messages:
                icon = QMessageBox.Warning if invalid_count > 0 else QMessageBox.Information
                QMessageBox(icon, "Data Processing Notice", "\n".join(messages), QMessageBox.Ok, self).exec_()

        except Exception as e:
            QMessageBox.critical(self, "Sleep-Stage Import Failed", f"Error: {str(e)}")
            self.status_label.setText("Status: sleep-stage import failed | model loaded")
            self.info_label.setText(f"Error: {str(e)}")



    def eventFilter(self, obj, event):
        if obj is self.plot_widget.viewport() and event.type() == QEvent.Wheel:
            if self.current_data is None or not self.time_slider.isEnabled():
                return True

            angle_delta = event.angleDelta()
            horizontal = angle_delta.x()
            vertical = angle_delta.y()
            if horizontal == 0 and vertical == 0:
                pixel_delta = event.pixelDelta()
                horizontal = pixel_delta.x()
                vertical = pixel_delta.y()

            if abs(horizontal) > abs(vertical):
                primary_delta = horizontal
                direction = 1 if primary_delta > 0 else -1
            else:
                primary_delta = vertical
                # Conventional vertical wheel: up = earlier/left, down = later/right.
                direction = -1 if primary_delta > 0 else 1

            if primary_delta != 0:
                steps = max(1, int(round(abs(primary_delta) / 120.0)))
                self.move_time_window(direction * steps)
            event.accept()
            return True
        return super().eventFilter(obj, event)

    def add_manual_spike_at_pending_time(self):
        if self.current_data is None or self.total_samples == 0:
            QMessageBox.warning(self, "Cannot Add", "Please import or load EEG data first.")
            return

        if self.pending_manual_spike_time is None:
            QMessageBox.information(self, "No Position Selected", "Click a blank waveform location first, then press A to add the spike.")
            return

        spike_time = self.pending_manual_spike_time
        self.pending_manual_spike_time = None
        self.add_manual_spike(spike_time)

    def move_time_window(self, direction):
        if self.current_data is None or not self.time_slider.isEnabled():
            return
        step = int(round(self.window_duration * self.fs))
        target = self.time_slider.value() + int(direction) * step
        target = max(self.time_slider.minimum(), min(target, self.time_slider.maximum()))
        self.time_slider.setValue(target)
        self.plot_widget.setFocus()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.adjust_amplitude_scale(0.2)
        elif event.key() == Qt.Key_Minus:
            self.adjust_amplitude_scale(-0.2)
        elif event.key() == Qt.Key_Left:
            self.move_time_window(-1)
            event.accept()
        elif event.key() == Qt.Key_Right:
            self.move_time_window(1)
            event.accept()
        elif event.key() == Qt.Key_Down:
            self.reject_current_annotation()
            event.accept()
        elif event.key() == Qt.Key_Up:
            self.confirm_current_annotation()
            event.accept()
        elif event.key() == Qt.Key_A:
            self.add_manual_spike_at_pending_time()
            event.accept()
        elif event.key() == Qt.Key_Delete:
            self.delete_selected_spikes()
            event.accept()
        else:
            super().keyPressEvent(event)

    def adjust_amplitude_scale(self, delta):
        self.amplitude_scale = max(0.1, min(10.0, self.amplitude_scale + float(delta)))
        if hasattr(self, 'amp_scale_label'):
            self.amp_scale_label.setText(f"{self.amplitude_scale:.1f}x")
        if self.current_data is not None:
            self.update_plot_by_slider(self.time_slider.value())
            self.plot_widget.setFocus()

    def reset_amplitude_scale(self):
        self.amplitude_scale = 1.0
        if hasattr(self, 'amp_scale_label'):
            self.amp_scale_label.setText(f"{self.amplitude_scale:.1f}x")
        if self.current_data is not None:
            self.update_plot_by_slider(self.time_slider.value())
            self.plot_widget.setFocus()

    def get_prediction_annotation_times(self):
        times = [float(t) for t, _ in self.high_prob_spikes] + [float(t) for t, _ in self.low_prob_spikes]
        return sorted(set(round(t, 3) for t in times))

    def jump_to_prediction_annotation(self, direction):
        annotation_times = self.get_prediction_annotation_times()
        if not annotation_times or self.current_data is None:
            return

        if not hasattr(self, 'current_annotation_nav_index'):
            self.current_annotation_nav_index = None

        if self.current_annotation_nav_index is None or self.current_annotation_nav_index >= len(annotation_times):
            current_center = self.time_slider.value() / self.fs + self.window_duration / 2
            self.current_annotation_nav_index = min(
                range(len(annotation_times)),
                key=lambda i: abs(annotation_times[i] - current_center)
            )
        else:
            self.current_annotation_nav_index = (self.current_annotation_nav_index + direction) % len(annotation_times)

        target_time = annotation_times[self.current_annotation_nav_index]
        target_start = int((target_time - self.window_duration / 2) * self.fs)
        target_start = max(0, min(target_start, self.time_slider.maximum()))
        self.time_slider.setValue(target_start)
        self.update_plot_by_slider(target_start)
        self.plot_widget.setFocus()



    def get_current_annotation_time(self):
        annotation_times = self.get_prediction_annotation_times()
        if not annotation_times or self.current_data is None:
            return None

        if (
            self.current_annotation_nav_index is None
            or self.current_annotation_nav_index >= len(annotation_times)
        ):
            current_center = self.time_slider.value() / self.fs + self.window_duration / 2
            self.current_annotation_nav_index = min(
                range(len(annotation_times)),
                key=lambda i: abs(annotation_times[i] - current_center)
            )

        return annotation_times[self.current_annotation_nav_index]

    def remove_annotation_at_time(self, spike_time):
        removed = False
        for spike_list in (self.high_prob_spikes, self.low_prob_spikes):
            for i, (t, prob) in enumerate(list(spike_list)):
                if abs(float(t) - float(spike_time)) <= 0.05:
                    del spike_list[i]
                    removed = True
                    break
            if removed:
                break

        if not removed:
            return False

        self.selected_spike_times.discard(self._spike_key(spike_time))
        self.current_annotation_nav_index = None
        self.update_plot_by_slider(self.time_slider.value())
        self.recalculate_swi()
        self.status_label.setText(
            f"Status: rejected/deleted annotation at {spike_time:.3f} s | purple {len(self.high_prob_spikes)}, orange {len(self.low_prob_spikes)}"
        )
        self.save_spike_label_autosave("remove_annotation")
        return True

    def confirm_annotation_at_time(self, spike_time):
        for i, (t, prob) in enumerate(list(self.low_prob_spikes)):
            if abs(float(t) - float(spike_time)) <= 0.05:
                del self.low_prob_spikes[i]
                self.high_prob_spikes.append((float(t), float(prob)))
                self.high_prob_spikes.sort(key=lambda x: x[0])
                self.current_annotation_nav_index = None
                self.update_plot_by_slider(self.time_slider.value())
                self.recalculate_swi()
                self.status_label.setText(
                    f"Status: converted annotation at {spike_time:.3f} s to purple | purple {len(self.high_prob_spikes)}, orange {len(self.low_prob_spikes)}"
                )
                self.save_spike_label_autosave("confirm_annotation")
                return True

        for t, prob in self.high_prob_spikes:
            if abs(float(t) - float(spike_time)) <= 0.05:
                self.status_label.setText(f"Status: {spike_time:.3f} s is already a purple annotation")
                return True

        return False

    def reject_current_annotation(self):
        spike_time = self.get_current_annotation_time()
        if spike_time is None:
            QMessageBox.information(self, "No Annotation", "There is no annotation available to reject.")
            return
        self.remove_annotation_at_time(spike_time)

    def confirm_current_annotation(self):
        spike_time = self.get_current_annotation_time()
        if spike_time is None:
            QMessageBox.information(self, "No Annotation", "There is no annotation available to convert to purple.")
            return
        self.confirm_annotation_at_time(spike_time)

    def zoom_y_axis(self, scale_factor):
        if self.plot_widget is None:
            return
        y_min, y_max = self.plot_widget.viewRange()[1]
        center = (y_min + y_max) / 2
        half_range = (y_max - y_min) / 2
        new_half_range = half_range / scale_factor
        new_y_min = center - new_half_range
        new_y_max = center + new_half_range
        self.plot_widget.setYRange(new_y_min, new_y_max, padding=0)
        self.current_y_scale *= scale_factor
        scaled_unit = self.original_y_unit_scale * self.current_y_scale
        self.plot_widget.setLabel('left', '')


    def _spike_key(self, spike_time):
        return round(float(spike_time), 3)

    def _find_spike_at_plot_time(self, click_x):
        for region, spike_time in self.region_spike_map.items():
            region_x = region.getRegion()
            if region_x[0] <= click_x <= region_x[1]:
                return float(spike_time)
        return None

    def _toggle_selected_spike(self, spike_time):
        key = self._spike_key(spike_time)
        if key in self.selected_spike_times:
            self.selected_spike_times.remove(key)
        else:
            self.selected_spike_times.add(key)
        self.pending_manual_spike_time = None
        self.update_plot_by_slider(self.time_slider.value())
        self.status_label.setText(
            f"Status: selected {len(self.selected_spike_times)} labels | right-click or press Delete to remove; click blank space then press A to add"
        )

    def _set_pending_manual_spike(self, spike_time):
        total_duration = self.total_samples / self.fs if self.fs else 0
        self.pending_manual_spike_time = round(max(0.0, min(float(spike_time), total_duration)), 3)
        self.update_plot_by_slider(self.time_slider.value())
        self.status_label.setText(
            f"Status: pending add at {self.pending_manual_spike_time:.3f} s | press A to add a purple label"
        )

    def delete_selected_spikes(self):
        if not self.selected_spike_times:
            QMessageBox.information(self, "No Selected Labels", "Please click one or more orange/purple label regions first.")
            return

        selected_keys = set(self.selected_spike_times)

        def keep_unselected(spike):
            return self._spike_key(spike[0]) not in selected_keys

        old_high = len(self.high_prob_spikes)
        old_low = len(self.low_prob_spikes)
        self.high_prob_spikes = [s for s in self.high_prob_spikes if keep_unselected(s)]
        self.low_prob_spikes = [s for s in self.low_prob_spikes if keep_unselected(s)]
        deleted_count = (old_high - len(self.high_prob_spikes)) + (old_low - len(self.low_prob_spikes))

        if deleted_count == 0:
            self.selected_spike_times.clear()
            self.update_plot_by_slider(self.time_slider.value())
            self.status_label.setText("Status: selected labels no longer exist")
            return

        self.selected_spike_times.clear()
        self.current_annotation_nav_index = None
        self.recalculate_swi()
        self.update_plot_by_slider(self.time_slider.value())
        self.status_label.setText(
            f"Status: deleted {deleted_count} labels | purple {len(self.high_prob_spikes)}, orange {len(self.low_prob_spikes)}"
        )
        self.save_spike_label_autosave("delete_selected")

    def show_spike_context_menu(self, event, click_x):
        clicked_spike = self._find_spike_at_plot_time(click_x)
        if clicked_spike is not None and self._spike_key(clicked_spike) not in self.selected_spike_times:
            self.selected_spike_times.add(self._spike_key(clicked_spike))
            self.update_plot_by_slider(self.time_slider.value())

        menu = QMenu(self)
        delete_action = menu.addAction(f"Delete Selected Labels ({len(self.selected_spike_times)})")
        delete_action.setEnabled(bool(self.selected_spike_times))
        chosen = menu.exec_(event.globalPos())
        if chosen == delete_action:
            self.delete_selected_spikes()

    def on_plot_mouse_press(self, event):
        if event.button() == Qt.RightButton:
            scene_pos = self.plot_widget.mapToScene(event.pos())
            click_x = self.plot_widget.plotItem.vb.mapSceneToView(scene_pos).x()
            self.show_spike_context_menu(event, click_x)
            event.accept()
            return

        self._original_plot_mouse_press_event(event)

    def on_plot_click(self, event):
        click_x = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos()).x()

        if self.process_step != 3:
            return

        if event.button() == Qt.LeftButton:
            clicked_spike = self._find_spike_at_plot_time(click_x)
            if clicked_spike is not None:
                self._toggle_selected_spike(clicked_spike)
            else:
                self._set_pending_manual_spike(click_x)
            return

    def toggle_manual_label_mode(self):
        self.status_label.setText("Status: manual label mode active. Click a blank waveform location, then press A to confirm.")
        self.plot_widget.setFocus()
    def add_manual_spike(self, spike_time):
        if self.current_data is None or self.total_samples == 0:
            QMessageBox.warning(self, "Cannot Add", "Please import or load EEG data first.")
            return

        total_duration = self.total_samples / self.fs
        spike_time = round(max(0.0, min(float(spike_time), total_duration)), 3)
        existing_times = [float(t) for t, _ in self.high_prob_spikes] + [float(t) for t, _ in self.low_prob_spikes]
        if any(abs(spike_time - t) <= 0.05 for t in existing_times):
            QMessageBox.information(self, "Already Exists", f"An annotation already exists near {spike_time:.3f} s.")
            return

        self.high_prob_spikes.append((spike_time, 1.0))
        self.high_prob_spikes.sort(key=lambda x: x[0])
        self.selected_spike_times.clear()
        annotation_times = self.get_prediction_annotation_times()
        self.current_annotation_nav_index = annotation_times.index(spike_time) if spike_time in annotation_times else None

        if self.window_center_times:
            nearest_time = min(self.window_center_times, key=lambda t: abs(t - spike_time))
            self.window_pred_map[nearest_time] = 1

        self.update_plot_by_slider(self.time_slider.value())
        self.save_spike_btn.setEnabled(True)
        self.recalc_swi_btn.setEnabled(True)
        self.plot_brain_btn.setEnabled(True)
        all_spike_times = [t for t, _ in self.high_prob_spikes] + [t for t, _ in self.low_prob_spikes]
        self.swi_value, self.spike_seconds, self.total_seconds = calculate_swi(all_spike_times, total_duration)
        self.swi_label.setText(
            f"Spike-Wave Index (SWI)\n"
            f"1. SWI: {self.swi_value}%\n"
            f"2. High-probability spikes: {len(self.high_prob_spikes)}\n"
            f"3. Low-probability spikes: {len(self.low_prob_spikes)}\n"
            f"4. Seconds containing spikes: {self.spike_seconds} s\n"
            f"5. Total seconds: {self.total_seconds} s\n"
            f"6. Original duration: {total_duration:.2f} s"
        )
        self.status_label.setText(
            f"Status: manually added purple annotation at {spike_time:.3f} s | purple {len(self.high_prob_spikes)}, orange {len(self.low_prob_spikes)}"
        )
        self.info_label.setText(
            f"Manually added purple annotation: {spike_time:.3f} s\n"
            f"Purple annotations: {len(self.high_prob_spikes)} | orange annotations: {len(self.low_prob_spikes)}\n"
            f"Click a blank waveform location to choose another position, then press A to add it; click existing purple/orange labels to multi-select and delete."
        )
        self.save_spike_label_autosave("add_manual")

    def confirm_low_spike(self, spike_time):
        for i, (t, prob) in enumerate(self.low_prob_spikes):
            if abs(t - spike_time) < 1e-3:
                confirmed_spike = (t, prob)
                del self.low_prob_spikes[i]
                break
        else:
            return

        self.high_prob_spikes.append(confirmed_spike)
        self.high_prob_spikes.sort(key=lambda x: x[0])

        self.update_plot_by_slider(self.time_slider.value())
        self.save_spike_label_autosave("confirm_low_spike")

        QMessageBox.information(self, "Done", f"Confirmed the spike at {spike_time:.3f} s and changed it to a purple region.")

    def reject_low_spike(self, spike_time):
        for i, (t, prob) in enumerate(self.low_prob_spikes):
            if abs(t - spike_time) < 1e-3:
                del self.low_prob_spikes[i]
                break
        else:
            return

        self.update_plot_by_slider(self.time_slider.value())
        self.save_spike_label_autosave("reject_low_spike")

        QMessageBox.information(self, "Done", f"Rejected the spike at {spike_time:.3f} s and removed the orange region.")

    def recalculate_swi(self):
        if self.total_samples == 0:
            QMessageBox.warning(self, "Calculation Failed", "No data available for SWI calculation.")
            return

        all_spike_times = [t for t, _ in self.high_prob_spikes] + [t for t, _ in self.low_prob_spikes]
        total_duration = self.total_samples / self.fs

        self.swi_value, self.spike_seconds, self.total_seconds = calculate_swi(
            all_spike_times, total_duration
        )

        self.swi_label.setText(
            f"Spike-Wave Index (SWI)\n"
            f"1. SWI: {self.swi_value}%\n"
            f"2. High-probability spikes: {len(self.high_prob_spikes)}\n"
            f"3. Low-probability spikes: {len(self.low_prob_spikes)}\n"
            f"4. Seconds containing spikes: {self.spike_seconds} s\n"
            f"5. Total seconds: {self.total_seconds} s\n"
            f"6. Original duration: {total_duration:.2f} s"
        )

        self.status_label.setText(
            f"Status: SWI recalculated | high-probability spikes {len(self.high_prob_spikes)}, low-probability spikes {len(self.low_prob_spikes)}"
        )

    def init_model(self):
        try:
            self.model = FusionModel(
                num_channels=16,
                time_steps=self.input_window_size,
                num_classes=2
            ).to(self.device)

            checkpoint = torch.load(self.model_weight_path, map_location=self.device, weights_only=True)
            if "module." in list(checkpoint.keys())[0]:
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            self.model.load_state_dict(checkpoint)

            self.model.eval()
            self.status_label.setText("Status: waiting for data import | model loaded")

        except FileNotFoundError:
            QMessageBox.critical(self, "Model Load Failed", f"Weight file not found.\nPath: {self.model_weight_path}")
            self.status_label.setText("Status: waiting for data import | model load failed (missing weights)")
            self.spike_detect_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Model Load Failed", f"Error: {str(e)}\nThe weights may not match the model architecture.")
            self.status_label.setText("Status: waiting for data import | model load failed (architecture mismatch)")
            self.spike_detect_btn.setEnabled(False)

    def import_raw_data(self):
        self.txt_raw_data = None
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Raw EEG Data (TXT/SET)", os.getcwd(), "EEG files (*.txt *.set);;All files (*.*)"
        )

        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Tip", "No valid file was selected, or the file does not exist.")
            return

        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == ".txt":
                warm_up_matlab_engine_async()
                all_lines = []
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_lines.append(line)
                total_rows = len(all_lines)
                if total_rows == 0:
                    raise ValueError("TXT file is empty; no valid data were found.")

                dialog = TimeRowRangeDialog(
                    total_rows=total_rows,
                    fs=self.fs,
                    parent=self
                )
                if dialog.exec_() != QDialog.Accepted:
                    return

                row_start = dialog.selected_start_row
                row_end = dialog.selected_end_row

                data_list = []
                selected_lines = all_lines[row_start - 1:row_end]

                over_16_count = 0
                invalid_count = 0
                over_16_samples = []
                invalid_samples = []

                for line_num, line in enumerate(selected_lines, start=row_start):
                    values = [float(val) for val in line.replace(',', ' ').split()]
                    if len(values) > 16:
                        over_16_count += 1
                        over_16_samples.append(line_num)
                        values = values[:16]
                    elif len(values) != 16:
                        invalid_count += 1
                        invalid_samples.append(line_num)
                        continue
                    data_list.append(values)

                messages = []
                if over_16_count > 0:
                    if over_16_count <= 5:
                        rows_str = ", ".join(map(str, over_16_samples))
                    else:
                        rows_str = ", ".join(map(str, over_16_samples[:5])) + f" and {over_16_count} rows total"
                    messages.append(f"Detected {over_16_count} rows with more than 16 channels: {rows_str}")
                    messages.append("Only the first 16 channels were kept for analysis.")

                if invalid_count > 0:
                    if invalid_count <= 5:
                        rows_str = ", ".join(map(str, invalid_samples))
                    else:
                        rows_str = ", ".join(map(str, invalid_samples[:5])) + f" and {invalid_count} rows total"
                    messages.append(f"Found {invalid_count} rows with fewer than 16 channels: {rows_str}")
                    messages.append("These rows were skipped and excluded from analysis.")

                if messages:
                    icon = QMessageBox.Warning if invalid_count > 0 else QMessageBox.Information
                    QMessageBox(
                        icon,
                        "Data Processing Notice",
                        "\n".join(messages),
                        QMessageBox.Ok,
                        self
                    ).exec_()

                data_np = np.array(data_list).T
                if data_np.shape[0] > 16:
                    data_np = data_np[:16, :]
                    QMessageBox.information(
                        self,
                        "Data Processing Notice",
                        "The total channel count is greater than 16. Only the first 16 channels were kept.",
                        QMessageBox.Ok
                    )
                elif data_np.shape[0] != 16:
                    raise ValueError(f"Invalid channel count: expected 16, got {data_np.shape[0]}.")

                info = mne.create_info(
                    ch_names=self.channel_names,
                    sfreq=self.fs,
                    ch_types=['eeg']*16
                )
                self.raw = mne.io.RawArray(data_np, info)

            elif file_ext == ".set":
                self.raw = mne.io.read_raw_eeglab(file_path, preload=True)
            else:
                raise ValueError(f"Unsupported format: {file_ext} (TXT/SET only)")

            self.current_data = self.raw.get_data()
            self.current_data = (self.current_data - np.mean(self.current_data, axis=1, keepdims=True)) / \
                                (np.std(self.current_data, axis=1, keepdims=True) + 1e-6)

            self.process_step = 1
            self.total_samples = self.current_data.shape[1]
            max_slider_value = max(0, self.total_samples - self.samples_per_window)
            self.time_slider.setMaximum(max_slider_value)
            self.time_slider.setEnabled(True)

            self.update_plot_by_slider(0)
            self.file_path = file_path
            self.eeg_source_path = file_path
            self.eeg_source_intervals = []
            self.eeg_source_invert = False
            self.last_loaded_spike_path = None

            self.high_prob_spikes.clear()
            self.low_prob_spikes.clear()
            self.current_annotation_nav_index = None
            self.region_spike_map.clear()
            self.window_center_times.clear()
            self.window_pred_map.clear()  # 重置预测映射
            self.swi_value = 0.0
            self.spike_seconds = 0
            self.total_seconds = 0
            self.swi_label.setText("Spike-Wave Index (SWI)\nWaiting for detection or loaded results...")
            self.save_spike_btn.setEnabled(False)
            self.recalc_swi_btn.setEnabled(False)
            self.plot_brain_btn.setEnabled(False)

            file_type = "TXT" if file_ext == ".txt" else "SET"
            self.status_label.setText(f"Status: imported raw {file_type} data | model loaded")
            self.info_label.setText(
                f"Raw {file_type} data imported successfully.\n"
                f"File: {os.path.basename(file_path)}\n"
                f"Channels: 16 | Sampling rate: {self.fs} Hz\n"
                f"Data duration: {self.total_samples/self.fs:.2f} s\n"
                f"Next step: click Run Preprocessing"
            )
            self.preprocess_btn.setEnabled(True)
            self.spike_detect_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Error reading raw file:\n{str(e)}")
            self.status_label.setText("Status: raw data import failed | model loaded")
            self.info_label.setText(f"Error: {str(e)}")

    def import_processed_data(self):
        self.txt_raw_data = None
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Preprocessed EEG Data (.set)", os.getcwd(), "EEG files (*.set);;All files (*.*)"
        )

        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Tip", "No valid file was selected, or the file does not exist.")
            return

        try:
            self.raw = mne.io.read_raw_eeglab(file_path, preload=True)
            self.current_data = self.raw.get_data()
            self.current_data = (self.current_data - np.mean(self.current_data, axis=1, keepdims=True)) / \
                                (np.std(self.current_data, axis=1, keepdims=True) + 1e-6)

            self.process_step = 2
            self.file_path = file_path
            self.eeg_source_path = file_path
            self.eeg_source_intervals = []
            self.eeg_source_invert = False
            self.last_loaded_spike_path = None
            self.total_samples = self.current_data.shape[1]
            max_slider_value = max(0, self.total_samples - self.samples_per_window)
            self.time_slider.setMaximum(max_slider_value)
            self.time_slider.setEnabled(True)

            self.update_plot_by_slider(0)

            self.high_prob_spikes.clear()
            self.low_prob_spikes.clear()
            self.current_annotation_nav_index = None
            self.region_spike_map.clear()
            self.window_center_times.clear()
            self.window_pred_map.clear()  # 重置预测映射
            self.swi_value = 0.0
            self.spike_seconds = 0
            self.total_seconds = 0
            self.swi_label.setText("Spike-Wave Index (SWI)\nWaiting for detection or loaded results...")
            self.save_spike_btn.setEnabled(False)
            self.recalc_swi_btn.setEnabled(False)
            self.plot_brain_btn.setEnabled(False)

            self.status_label.setText("Status: imported preprocessed data | model loaded")
            self.info_label.setText(
                f"Preprocessed data imported successfully.\n"
                f"File: {os.path.basename(file_path)}\n"
                f"Channels: 16 | Sampling rate: {self.fs} Hz\n"
                f"Data duration: {self.total_samples/self.fs:.2f} s\n"
                f"Next step: click Detect Spikes"
            )
            self.preprocess_btn.setEnabled(False)
            self.spike_detect_btn.setEnabled(True if self.model is not None else False)

        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Error reading preprocessed file:\n{str(e)}")
            self.status_label.setText("Status: preprocessed data import failed | model loaded")
            self.info_label.setText(f"Error: {str(e)}")

    def run_preprocessing(self):
        if self.process_step != 1 or not self.file_path:
            QMessageBox.warning(self, "Tip", "Please import raw data first.")
            return

        progress = QProgressDialog("Calling MATLAB for preprocessing...", "Cancel", 0, 4, self)
        progress.setWindowTitle("MATLAB Preprocessing")
        progress.setWindowModality(2)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def run_matlab_preprocess_responsive(file_path, fs, save_dir):
            """Run the blocking MATLAB Engine call without freezing the Qt GUI."""
            progress.setRange(0, 0)
            QApplication.processEvents()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as matlab_executor:
                    matlab_future = matlab_executor.submit(
                        call_matlab_preprocess,
                        file_path,
                        fs,
                        save_dir,
                    )
                    while not matlab_future.done():
                        QApplication.processEvents()
                        time.sleep(0.03)
                    return matlab_future.result()
            finally:
                progress.setRange(0, 4)
                QApplication.processEvents()

        try:
            progress.setValue(1)
            progress.setLabelText("Step 1/4: preparing input data...")
            QApplication.processEvents()

            if os.path.splitext(self.file_path)[1].lower() == ".txt":
                progress.setLabelText("Step 1/4: exporting temporary SET file...")
                QApplication.processEvents()
                temp_set_dir = os.path.join(INTERFACE_DATA_DIR, "temp_set")
                os.makedirs(temp_set_dir, exist_ok=True)
                temp_set_name = os.path.splitext(os.path.basename(self.file_path))[0] + "_temp.set"
                temp_set_path = os.path.join(temp_set_dir, temp_set_name)

                if self.raw is not None:
                    raw_data = self.raw.get_data()
                    info = self.raw.info
                elif self.txt_raw_data is not None:
                    raw_data = self.txt_raw_data
                    info = mne.create_info(
                        ch_names=self.channel_names,
                        sfreq=self.fs,
                        ch_types=['eeg'] * len(self.channel_names),
                    )
                else:
                    raise RuntimeError("The imported raw TXT data are unavailable. Please import the TXT again.")
                raw_data_negated = np.negative(np.asarray(raw_data, dtype=np.float32))
                processed_raw = mne.io.RawArray(raw_data_negated, info)

                mne.export.export_raw(temp_set_path, processed_raw, fmt='eeglab', overwrite=True)
                self.txt_raw_data = None
                del raw_data, raw_data_negated, processed_raw
                progress.setValue(2)
                progress.setLabelText(
                    "Step 2/4: MATLAB ready; filtering..."
                    if matlab_engine_is_ready()
                    else "Step 2/4: waiting for background MATLAB startup, then filtering..."
                )
                QApplication.processEvents()
                full_set_path = run_matlab_preprocess_responsive(
                    temp_set_path,
                    self.fs,
                    os.path.join(INTERFACE_DATA_DIR, "filter"),
                )
            else:
                progress.setValue(2)
                progress.setLabelText(
                    "Step 2/4: MATLAB ready; filtering..."
                    if matlab_engine_is_ready()
                    else "Step 2/4: starting MATLAB, then filtering..."
                )
                QApplication.processEvents()
                full_set_path = run_matlab_preprocess_responsive(
                    self.file_path,
                    self.fs,
                    os.path.join(INTERFACE_DATA_DIR, "filter"),
                )

            progress.setValue(3)
            progress.setLabelText("Step 3/4: loading preprocessed data...")
            QApplication.processEvents()

            self.current_data, loaded_fs = load_eeg_from_source(
                full_set_path,
                [],
                False,
                self.channel_names,
                self.fs,
            )
            self.fs = loaded_fs
            self.samples_per_window = int(self.window_duration * self.fs)
            self.eeg_source_path = full_set_path
            self.eeg_source_intervals = []
            self.eeg_source_invert = False

            progress.setValue(4)
            progress.setLabelText("Step 4/4: preprocessing complete...")
            QApplication.processEvents()

            self.process_step = 2
            self.total_samples = self.current_data.shape[1]
            max_slider_value = max(0, self.total_samples - self.samples_per_window)
            self.time_slider.setMaximum(max_slider_value)

            self.update_plot_by_slider(0)

            self.high_prob_spikes.clear()
            self.low_prob_spikes.clear()
            self.current_annotation_nav_index = None
            self.region_spike_map.clear()
            self.window_center_times.clear()
            self.window_pred_map.clear()  # 重置预测映射
            self.swi_value = 0.0
            self.spike_seconds = 0
            self.total_seconds = 0
            self.swi_label.setText("Spike-Wave Index (SWI)\nWaiting for detection or loaded results...")
            self.save_spike_btn.setEnabled(False)
            self.recalc_swi_btn.setEnabled(False)
            self.plot_brain_btn.setEnabled(False)

            self.status_label.setText("Status: preprocessing complete | model loaded")
            self.info_label.setText(
                f"MATLAB preprocessing complete.\n"
                f"File: {os.path.basename(full_set_path)}\n"
                f"Data shape: {self.current_data.shape[0]} channels x {self.current_data.shape[1]} samples\n"
                f"Next step: click Detect Spikes"
            )
            self.spike_detect_btn.setEnabled(True if self.model is not None else False)

        except Exception as e:
            QMessageBox.critical(self, "MATLAB Preprocessing Failed", f"Error: {str(e)}")
            self.status_label.setText("Status: preprocessing failed | model loaded")
            self.info_label.setText(f"Error: {str(e)}")
        finally:
            progress.close()

    def run_spike_detection(self):
        if self.process_step != 2 or self.model is None:
            QMessageBox.warning(self, "Requirements Not Met", "Please finish preprocessing and make sure the model loaded successfully.")
            return

        progress = QProgressDialog("Running spike detection...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Spike Detection")
        progress.setWindowModality(2)
        progress.setValue(0)
        progress.setMinimumDuration(0)
        detection_completed = False

        try:
            start_time = time.time()
            data = self.current_data
            n_channels, total_samples = data.shape

            max_start = total_samples - self.input_window_size
            if max_start <= 0:
                raise Exception(f"Data length is shorter than {self.window_sec} s (at least {self.input_window_size} samples required).")

            total_duration = total_samples / self.fs
            temp_spikes = []
            window_step = self.half_window
            total_windows = len(range(0, max_start + 1, window_step))
            prob_threshold = 0.8

            # 重置窗口中心时间和预测映射
            self.window_center_times = []
            self.window_pred_map = {}

            self.artefact_idx.clear()
            if self.device.type == "cuda":
                batch_size = INFERENCE_BATCH_SIZE
            else:
                batch_size = CPU_INFERENCE_BATCH_SIZE
            batch_features = []
            batch_times = []
            batch_raw_times = []

            def run_inference_batch():
                if not batch_features:
                    return
                input_tensor = torch.tensor(np.stack(batch_features, axis=0), dtype=torch.float32).to(self.device)
                with torch.inference_mode():
                    output = self.model(input_tensor)
                    probs = F.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
                    preds = torch.argmax(output, dim=1).detach().cpu().numpy()

                for mid_time_rounded, mid_time, pred, prob in zip(batch_times, batch_raw_times, preds, probs):
                    pred = int(pred)
                    prob = float(prob)
                    self.window_pred_map[mid_time_rounded] = pred
                    if pred == 1:
                        temp_spikes.append((round(mid_time, 3), round(prob, 3)))

                batch_features.clear()
                batch_times.clear()
                batch_raw_times.clear()
                QApplication.processEvents()

            for idx, start in enumerate(range(0, max_start + 1, window_step)):
                if progress.wasCanceled():
                    raise Exception("Spike detection canceled by user")

                progress.setValue(int((idx / total_windows) * 100))
                QApplication.processEvents()

                end = start + self.input_window_size
                seg = data[:, start:end]

                seg_max_abs = np.max(np.abs(seg))
                if seg_max_abs > self.artefact_threshold:
                    self.artefact_idx.append(idx)
                    continue

                # 记录窗口中心时间（用于后续脑地形图分析）
                mid_time = (start + self.half_window) / self.fs
                mid_time_rounded = round(mid_time, 3)
                self.window_center_times.append(mid_time_rounded)

                feat = extract_multi_channel_features(seg)
                batch_features.append(feat)
                batch_times.append(mid_time_rounded)
                batch_raw_times.append(mid_time)

                if len(batch_features) >= batch_size:
                    run_inference_batch()

            run_inference_batch()

            def merge_spikes(spike_list):
                if not spike_list:
                    return []
                sorted_spikes = sorted(spike_list, key=lambda x: x[0])
                merged = [sorted_spikes[0]]
                for t, prob in sorted_spikes[1:]:
                    if t - merged[-1][0] < 0.1:
                        avg_t = (merged[-1][0] + t) / 2
                        avg_prob = (merged[-1][1] + prob) / 2
                        merged[-1] = (round(avg_t, 3), round(avg_prob, 3))
                    else:
                        merged.append((t, prob))
                return merged

            merged_spikes = merge_spikes(temp_spikes)

            self.high_prob_spikes = [s for s in merged_spikes if s[1] >= prob_threshold]
            self.low_prob_spikes = [s for s in merged_spikes if s[1] > 0.5 and s[1] < prob_threshold]

            all_spike_times = [t for t, _ in self.high_prob_spikes] + [t for t, _ in self.low_prob_spikes]
            self.swi_value, self.spike_seconds, self.total_seconds = calculate_swi(
                all_spike_times, total_duration
            )

            self.process_step = 3
            progress.setValue(100)
            detection_time = time.time() - start_time
            self.last_detection_elapsed_sec = detection_time
            self.update_long_term_record(
                **{
                    "棘波检测用时": self.format_duration(detection_time),
                }
            )

            self.update_plot_by_slider(self.time_slider.value())
            self.status_label.setText(
                f"Status: spike detection complete | high-probability {len(self.high_prob_spikes)}, low-probability {len(self.low_prob_spikes)}"
            )

            self.swi_label.setText(
                f"Spike-Wave Index (SWI)\n"
                f"1. SWI: {self.swi_value}%\n"
                f"2. High-probability spikes: {len(self.high_prob_spikes)}\n"
                f"3. Low-probability spikes: {len(self.low_prob_spikes)}\n"
                f"4. Seconds containing spikes: {self.spike_seconds} s\n"
                f"5. Total seconds: {self.total_seconds} s\n"
                f"6. Original duration: {total_duration:.2f} s"
            )

            self.save_spike_btn.setEnabled(True)
            self.recalc_swi_btn.setEnabled(True)
            self.plot_brain_btn.setEnabled(True)

            if self.artefact_idx:
                artefact_count = len(self.artefact_idx)
                self.info_label.setText(
                    f"Spike detection complete.\n"
                    f"Elapsed time: {self.seconds_to_hms(detection_time)}\n"
                    f"Detection windows: {total_windows} | artefact windows: {artefact_count} (skipped)\n"
                    f"Tip: click orange regions to confirm/reject; click Recalculate SWI after edits.\n"
                    f"Click View Spatial Distribution to inspect spatial distribution."
                )
            else:
                self.info_label.setText(
                    f"Spike detection complete.\n"
                    f"Elapsed time: {self.seconds_to_hms(detection_time)}\n"
                    f"Detection windows: {total_windows} | no artefact windows detected\n"
                    f"Tip: click orange regions to confirm/reject; click Recalculate SWI after edits.\n"
                    f"Click View Spatial Distribution to inspect spatial distribution."
                )
            detection_completed = True

        except Exception as e:
            if str(e) == "Spike detection canceled by user":
                QMessageBox.information(self, "Detection Canceled", "Spike detection was canceled.")
                self.status_label.setText("Status: spike detection canceled | model loaded")
            else:
                QMessageBox.critical(self, "Detection Failed", f"Error: {str(e)}")
                self.status_label.setText("Status: spike detection failed | model loaded")
            self.info_label.setText(f"Error: {str(e)}")
            self.recalc_swi_btn.setEnabled(False)
            self.plot_brain_btn.setEnabled(False)
        finally:
            progress.close()
            if detection_completed:
                self.maybe_restore_spike_label_autosave()

    def seconds_to_hms(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        else:
            return f"{minutes}m {seconds}s"

    def save_spike_click(self):
        save_path = self.get_result_path(AUTO_RESULT_DIR, "spike_detection")
        saved_path = self.save_current_spikes_to_path(save_path, show_dialog=False)
        if not saved_path:
            return
        updates = {"棘波检测结果": saved_path}
        if self.last_detection_elapsed_sec is not None:
            updates["棘波检测用时"] = self.format_duration(self.last_detection_elapsed_sec)
        self.update_long_term_record(**updates)
        self.clear_spike_label_autosave()
        QMessageBox.information(
            self,
            "Saved",
            f"Spike detection results saved to:\n{saved_path}\n\nRecord updated:\n{LONG_TERM_RECORD_CSV}"
        )


    def load_spike_click(self):
        self.txt_raw_data = None
        spike_data = load_spike_data()
        if not spike_data:
            return

        self.last_loaded_spike_path = spike_data.get("load_path")
        loaded_source_path = spike_data.get("source_file_path")
        if loaded_source_path:
            self.file_path = loaded_source_path
            self.eeg_source_path = loaded_source_path
            self.eeg_source_intervals = spike_data.get("source_intervals", [])
            self.eeg_source_invert = spike_data.get("source_invert", False)
        else:
            # A legacy result contains EEG samples but no reproducible source.
            self.eeg_source_path = None
            self.eeg_source_intervals = []
            self.eeg_source_invert = False
        if self.last_loaded_spike_path:
            match = re.match(r"(S\d{3})_", os.path.basename(self.last_loaded_spike_path), re.IGNORECASE)
            if match:
                self.subject_id = match.group(1).upper()
        self.high_prob_spikes = [tuple(x) for x in spike_data.get("high_prob_spikes", [])]
        self.low_prob_spikes = [tuple(x) for x in spike_data.get("low_prob_spikes", [])]
        if not self.high_prob_spikes and not self.low_prob_spikes:
            self.high_prob_spikes = [(t, 1.0) for t in spike_data["spike_times"]]
        self.high_prob_spikes.sort(key=lambda x: x[0])
        self.low_prob_spikes.sort(key=lambda x: x[0])
        self.current_annotation_nav_index = None
        self.current_data = spike_data["eeg_data"]
        self.fs = spike_data["fs"]
        self.samples_per_window = int(self.window_duration * self.fs)
        self.total_samples = self.current_data.shape[1]
        total_duration = self.total_samples / self.fs

        # 重新生成窗口中心时间和预测映射，用于SWI和脑图统计。
        max_start = self.total_samples - self.input_window_size
        self.window_center_times = []
        self.window_pred_map = {}
        if max_start > 0:
            window_step = self.half_window
            # Build all window labels at once.  The previous implementation did a
            # Python scan over every spike for every window (O(windows * spikes));
            # a long recording could therefore require billions of comparisons.
            starts = np.arange(0, max_start + 1, window_step, dtype=np.int64)
            center_times = np.round((starts + self.half_window) / self.fs, 3)
            spike_times_sorted = np.sort(np.asarray(
                [t for t, _ in self.high_prob_spikes] +
                [t for t, _ in self.low_prob_spikes],
                dtype=float
            ))

            if spike_times_sorted.size:
                insert_at = np.searchsorted(spike_times_sorted, center_times, side="left")
                nearest_distance = np.full(center_times.shape, np.inf, dtype=float)

                has_right = insert_at < spike_times_sorted.size
                nearest_distance[has_right] = np.abs(
                    spike_times_sorted[insert_at[has_right]] - center_times[has_right]
                )

                has_left = insert_at > 0
                nearest_distance[has_left] = np.minimum(
                    nearest_distance[has_left],
                    np.abs(spike_times_sorted[insert_at[has_left] - 1] - center_times[has_left])
                )
                window_predictions = (nearest_distance <= 0.5).astype(np.uint8)
            else:
                window_predictions = np.zeros(center_times.shape, dtype=np.uint8)

            self.window_center_times = center_times.tolist()
            self.window_pred_map = dict(zip(
                self.window_center_times,
                window_predictions.astype(int).tolist()
            ))

        all_spike_times = [t for t, _ in self.high_prob_spikes] + [t for t, _ in self.low_prob_spikes]
        self.swi_value, self.spike_seconds, self.total_seconds = calculate_swi(
            all_spike_times, total_duration
        )

        max_slider_value = max(0, self.total_samples - self.samples_per_window)
        self.time_slider.setMaximum(max_slider_value)
        self.time_slider.setEnabled(True)
        self.update_plot_by_slider(0)
        self.process_step = 3

        self.status_label.setText(
            f"Status: loaded spike results | purple {len(self.high_prob_spikes)}, orange {len(self.low_prob_spikes)}"
        )
        self.swi_label.setText(
            f"Spike-Wave Index (SWI)\n"
            f"1. SWI: {self.swi_value}%\n"
            f"2. High-probability spikes: {len(self.high_prob_spikes)}\n"
            f"3. Low-probability spikes: {len(self.low_prob_spikes)}\n"
            f"4. Seconds containing spikes: {self.spike_seconds} s\n"
            f"5. Total seconds: {self.total_seconds} s\n"
            f"6. Original duration: {total_duration:.2f} s\n"
            f"7. Save time: {spike_data['save_time']}"
        )

        self.save_spike_btn.setEnabled(True)
        self.recalc_swi_btn.setEnabled(True)
        self.plot_brain_btn.setEnabled(True)

        self.info_label.setText(
            f"Spike results loaded successfully.\n"
            f"File save time: {spike_data['save_time']}\n"
            f"Purple annotations: {len(self.high_prob_spikes)} | orange annotations: {len(self.low_prob_spikes)}\n"
            f"Data shape: {self.current_data.shape[0]} channels x {self.current_data.shape[1]} samples\n"
            f"Sampling rate: {self.fs} Hz\n"
            f"Click View Spatial Distribution to inspect spatial distribution."
        )
        self.maybe_restore_spike_label_autosave()


    def update_plot_by_slider(self, slider_value):
        if self.current_data is None:
            return

        start_sample = slider_value
        end_sample = start_sample + self.samples_per_window
        if end_sample > self.total_samples:
            end_sample = self.total_samples
            start_sample = max(0, end_sample - self.samples_per_window)

        plot_data = self.current_data[:, start_sample:end_sample]
        times = np.arange(start_sample, end_sample) / self.fs
        window_start = round(times[0], 2) if len(times) > 0 else 0
        window_end = round(times[-1], 2) if len(times) > 0 else self.window_duration

        self.plot_widget.clear()
        if self.plot_legend is not None:
            self.plot_legend.clear()
        self.region_spike_map.clear()
        self.plot_widget.setTitle(f"EEG Waveform | Time Range: {window_start}-{window_end} s (15-second window) | Amplitude {self.amplitude_scale:.1f}x")
        self.plot_widget.setMouseEnabled(x=True, y=True)

        # Plot scaling only needs the visible 15-second window.  Scanning a
        # multi-hour recording here made initial display and every slider move slow.
        data_std = np.std(plot_data, dtype=np.float64) if plot_data.size > 0 else 1
        if not np.isfinite(data_std) or data_std <= 0:
            data_std = 1
        offset = 5.0 * data_std
        highlight_channels = {4, 5, 12, 13}
        left_axis = self.plot_widget.getAxis('left')
        left_axis.setWidth(72)
        left_axis.setStyle(tickLength=0, tickTextOffset=8, tickFont=QFont("Arial", 16))
        y_positions = [(plot_data.shape[0] - 1 - i) * offset for i in range(plot_data.shape[0])]
        left_axis.setTicks([[tick for tick in zip(y_positions, self.channel_names[:plot_data.shape[0]])]])

        for i in range(plot_data.shape[0]):
            channel_data = plot_data[i] * self.amplitude_scale
            color = '#7B3294' if i in highlight_channels else '#0072B2'
            offset_data = channel_data + y_positions[i]

            self.plot_widget.plot(
                times, offset_data,
                label=self.channel_names[i],
                pen=pg.mkPen(color=color, width=2)
            )

        current_high_spikes = [s for s in self.high_prob_spikes if window_start <= s[0] <= window_end]
        for spike_time, _ in current_high_spikes:
            selected = self._spike_key(spike_time) in self.selected_spike_times
            region = LinearRegionItem(
                values=[spike_time - 0.1, spike_time + 0.1],
                brush=pg.mkBrush(color=(123, 50, 148, 110 if selected else 48)),
                movable=False
            )
            self.plot_widget.addItem(region)
            self.region_spike_map[region] = spike_time

        current_low_spikes = [s for s in self.low_prob_spikes if window_start <= s[0] <= window_end]
        for spike_time, _ in current_low_spikes:
            selected = self._spike_key(spike_time) in self.selected_spike_times
            region = LinearRegionItem(
                values=[spike_time - 0.1, spike_time + 0.1],
                brush=pg.mkBrush(color=(230, 159, 0, 110 if selected else 56)),
                movable=False
            )
            self.plot_widget.addItem(region)
            self.region_spike_map[region] = spike_time

        if self.pending_manual_spike_time is not None and window_start <= self.pending_manual_spike_time <= window_end:
            pending_line = pg.InfiniteLine(
                pos=self.pending_manual_spike_time,
                angle=90,
                pen=pg.mkPen(color=(123, 50, 148), width=2, style=Qt.DashLine),
                movable=False,
            )
            self.plot_widget.addItem(pending_line)
        if len(times) > 0:
            expected_end = times[0] + self.window_duration
            self.plot_widget.setXRange(times[0], expected_end, padding=0)

        if plot_data.size > 0:
            self.plot_widget.setYRange(-offset * 1.2, (plot_data.shape[0] - 1) * offset + offset * 1.2)
            self.current_y_scale = 1.0
            self.plot_widget.setLabel('left', '')

        bottom_axis = self.plot_widget.getAxis('bottom')
        bottom_axis.setStyle(tickFont=QFont("Arial", 10), tickTextOffset=4)
        self.plot_widget.setLabel('bottom', 'Time', 's', font=QFont("Arial", 10))

    def closeEvent(self, event):
        """Require explicit confirmation before closing the main window."""
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to close the program?\n\n"
            "Please make sure the current results and manual corrections have been saved.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            shutdown_matlab_engine()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 16))
    window = EEGAutoSaveProcessor()
    window.showMaximized()
    sys.exit(app.exec_())
