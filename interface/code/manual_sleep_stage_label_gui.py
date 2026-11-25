#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEG GUI: TFR-based manual sleep staging."""

from __future__ import annotations

import csv
import os
import sys
import warnings
from pathlib import Path

import mne
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton, QSizePolicy, QSlider,
    QSplitter, QVBoxLayout, QWidget,
)

warnings.filterwarnings("ignore", message=".*interactive.*deprecated.*")
mne.set_log_level("ERROR")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from plot_tfr_shenling import get_or_create_tfr_cache
except Exception as exc:
    get_or_create_tfr_cache = None
    TFR_IMPORT_ERROR = exc
else:
    TFR_IMPORT_ERROR = None

DEFAULT_EDF_DIR = r"D:\bai\BECT(Rolandic)_no_channel_name\20h测试数据"
SLEEP_CSV_DIR = Path(os.environ.get("MANUAL_SLEEP_LABEL_DIR", BASE_DIR / "manual_sleep_stage_labels"))
SEIZURE_CSV_DIR = BASE_DIR / "manual_seizure_time_csv"
SLEEP_CSV_DIR.mkdir(parents=True, exist_ok=True)
SEIZURE_CSV_DIR.mkdir(parents=True, exist_ok=True)

FREQ_RANGE = (0, 70)
V_MIN = -15
V_MAX = 15
DEFAULT_RAW_WINDOW_SEC = 15.0
RAW_WINDOW_OPTIONS = ["15", "30", "60", "120", "300"]
RAW_AMP_SCALE_OPTIONS = ["0.5x", "1x", "2x", "5x", "10x"]
MAX_RAW_POINTS = 5000
DEFAULT_SLEEP_BOX_SEC = 1800.0
SLEEP_EDGE_TOLERANCE_SEC = 120.0

SLEEP_STAGES = ["W", "NREM", "REM"]
SCOPES = ["Focal", "Generalized", "Uncertain"]
SEIZURE_TYPES = [
    "Focal motor seizure",
    "Focal non-motor seizure",
    "Focal to bilateral tonic-clonic",
    "Generalized tonic-clonic",
    "Absence seizure",
    "Myoclonic seizure",
    "Tonic seizure",
    "Clonic seizure",
    "Atonic seizure",
    "Epileptic spasm",
    "Electrographic-only seizure",
    "Suspected/uncertain seizure",
    "Other",
]
STAGE_COLORS = {
    "W": "#0072B2",
    "NREM": "#7B3294",
    "REM": "#EF6C00",
}

LEGACY_STAGE_MAP = {
    "清醒": "W",
    "W": "W",
    "N1": "NREM",
    "N2": "NREM",
    "N3": "NREM",
    "NREM": "NREM",
    "REM": "REM",
    "不确定": "NREM",
}


def sec2hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    return f"{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}:{int(sec % 60):02d}"


def sec2hm(x, pos=None):
    return f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}"


def pick_raw_channels(raw, n=16):
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    chs = [raw.ch_names[i] for i in picks]
    if not chs:
        chs = [ch for ch in raw.ch_names if "EEG" in ch.upper()]
    if not chs:
        chs = list(raw.ch_names)
    return chs[:n]


class SeizureLabelGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Manual Sleep Staging Tool")
        self.resize(1680, 720)
        self.setMinimumSize(1180, 620)
        self.setFocusPolicy(Qt.StrongFocus)

        self.current_edf = None
        self.raw = None
        self.raw_channels = []
        self.full_time = None
        self.freq_axis = None
        self.spec_db = None
        self.info = {}

        self.current_sleep_selection = None
        self.sleep_records = []
        self.sleep_patches = []
        self.pending_sleep_patch = None
        self.pending_sleep_edge = None
        self.pending_sleep_text = None
        self.sleep_dragging = False
        self.selected_sleep_idx = None

        self.seizure_records = []
        self.selected_seizure_idx = None
        self.current_seizure_start = None
        self.current_seizure_end = None
        self.current_scope = "Focal"
        self.current_seizure_type = "Focal motor seizure"
        self.raw_mark_mode = "start"
        self.raw_start = 0.0
        self.raw_window_sec = DEFAULT_RAW_WINDOW_SEC
        self.raw_amp_scale = 1.0
        self.slider_updating = False

        self.tfr_click_cid = None
        self.tfr_motion_cid = None
        self.tfr_release_cid = None
        self.raw_click_cid = None
        self.raw_scroll_cid = None

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        root.addWidget(self._build_toolbar())

        root.addWidget(self._build_tfr_panel(), stretch=1)

        self.status_label = QLabel("Ready: click Open EDF/SET to start")
        self.status_label.setStyleSheet("color:#555; padding:6px 12px; font-size:18px;")
        root.addWidget(self.status_label)

    def _build_toolbar(self):
        bar = QFrame()
        bar.setObjectName("toolbar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        self.btn_open = QPushButton("Open EDF/SET")
        self.btn_open.setObjectName("btn_open")
        self.btn_open.clicked.connect(self.on_open_edf)
        layout.addWidget(self.btn_open)

        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("file_label")
        self.file_label.setMinimumWidth(260)
        layout.addWidget(self.file_label, stretch=1)

        layout.addWidget(QLabel("Notes:"))
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("Optional")
        self.notes_edit.setMinimumWidth(180)
        layout.addWidget(self.notes_edit)

        return bar

    def _build_tfr_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        self.tfr_fig, self.tfr_ax = plt.subplots(figsize=(16, 7))
        self.tfr_fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.14)
        self.tfr_canvas = FigureCanvas(self.tfr_fig)
        self.tfr_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.tfr_canvas)
        layout.addWidget(NavToolbar(self.tfr_canvas, panel))
        return panel

    def _build_raw_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 4, 0)
        controls.addWidget(QLabel("Raw window:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(RAW_WINDOW_OPTIONS)
        self.window_combo.setCurrentText("15")
        self.window_combo.currentTextChanged.connect(self.on_window_changed)
        controls.addWidget(self.window_combo)

        controls.addWidget(QLabel("Amplitude:"))
        self.amp_combo = QComboBox()
        self.amp_combo.addItems(RAW_AMP_SCALE_OPTIONS)
        self.amp_combo.setCurrentText("1x")
        self.amp_combo.currentTextChanged.connect(self.on_amp_scale_changed)
        controls.addWidget(self.amp_combo)

        self.time_label = QLabel("00:00:00")
        self.time_label.setObjectName("time_label")
        self.time_label.setMinimumWidth(90)
        controls.addWidget(self.time_label)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self.on_slider_changed)
        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(145, 0, 45, 0)
        slider_row.addWidget(self.time_slider, stretch=1)
        layout.addLayout(slider_row)

        self.raw_fig, self.raw_ax = plt.subplots(figsize=(16, 4))
        self.raw_fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.14)
        self.raw_canvas = FigureCanvas(self.raw_fig)
        self.raw_canvas.setFocusPolicy(Qt.StrongFocus)
        self.raw_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.raw_canvas)
        layout.addWidget(NavToolbar(self.raw_canvas, panel))
        return panel

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background:#eef1f5; }
            QFrame#toolbar { background:#20242c; border-radius:8px; }
            QFrame#panel { background:white; border:1px solid #d6dbe3; border-radius:6px; }
            QLabel { color:#eef1f5; font-size:18px; }
            QLabel#file_label, QLabel#time_label { color:#d7dde8; }
            QFrame#panel QLabel { color:#334155; }
            QComboBox, QLineEdit {
                min-height:32px; padding:4px 10px; border:1px solid #c9cdd4;
                border-radius:4px; background:white;
                font-size:18px;
            }
            QSlider::groove:horizontal { height:6px; background:#cfd8dc; border-radius:3px; }
            QSlider::handle:horizontal { width:16px; margin:-5px 0; background:#1976D2; border-radius:8px; }
            QPushButton {
                min-height:34px; padding:7px 14px; border:none; border-radius:4px;
                font-weight:bold; color:white; background:#607D8B;
                font-size:18px;
            }
            QPushButton#btn_open { background:#1976D2; }
            QPushButton#btn_save { background:#7B3294; }
            QPushButton#btn_mark { background:#455A64; }
            QPushButton#btn_delete { background:#E69F00; }
            QPushButton#btn_undo { background:#0072B2; }
            QPushButton:disabled { background:#b8c0ca; color:#666; }
        """)

    @property
    def total_sec(self):
        if self.raw is not None and len(self.raw.times):
            return float(self.raw.times[-1])
        return float(self.info.get("total_rec_sec", 0.0))

    def on_open_edf(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EDF/SET File",
            DEFAULT_EDF_DIR if os.path.isdir(DEFAULT_EDF_DIR) else str(Path.home()),
            "EEG files (*.edf *.set);;EDF files (*.edf);;SET files (*.set);;All files (*.*)",
        )
        if path:
            self.load_edf(path)

    def load_edf(self, edf_path):
        if get_or_create_tfr_cache is None:
            QMessageBox.critical(self, "TFR Function Unavailable", f"Could not import plot_tfr_shenling.py:\n{TFR_IMPORT_ERROR}")
            return

        file_ext = Path(edf_path).suffix.lower()
        file_type = "SET" if file_ext == ".set" else "EDF"
        self.set_status(f"Reading {file_type} header...")
        QApplication.processEvents()
        try:
            if file_ext == ".set":
                raw = mne.io.read_raw_eeglab(edf_path, preload=False, verbose=False)
            elif file_ext == ".edf":
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            else:
                raise ValueError(f"Unsupported file format: {file_ext} (.edf/.set only)")
        except Exception as exc:
            QMessageBox.critical(self, "Open Failed", f"Failed to read {file_type}:\n{exc}")
            return

        self.raw = raw
        self.current_edf = edf_path
        self.file_label.setText(os.path.basename(edf_path))
        self.current_sleep_selection = None
        self.selected_sleep_idx = None
        self.pending_sleep_patch = None
        self.pending_sleep_edge = None
        self.pending_sleep_text = None
        self.sleep_dragging = False
        self.sleep_records = []
        self.set_status("Loading or computing the whole-night TFR...")
        QApplication.processEvents()
        try:
            self.full_time, self.freq_axis, self.spec_db, self.info = get_or_create_tfr_cache(edf_path)
        except Exception as exc:
            QMessageBox.critical(self, "TFR Load Failed", f"Failed to load/compute TFR:\n{exc}")
            return

        self._load_sleep_stages()
        self._draw_tfr()
        self.set_status(
            f"Loaded {os.path.basename(edf_path)} | {self.total_sec/3600:.2f}h | sleep stages: {len(self.sleep_records)}"
        )

    def _configure_slider(self):
        self.slider_updating = True
        max_start = max(0, int(round(self.total_sec - self.raw_window_sec)))
        self.time_slider.setRange(0, max_start)
        self.time_slider.setValue(0)
        self.time_slider.setEnabled(max_start > 0)
        self.slider_updating = False

    def on_window_changed(self, text):
        try:
            self.raw_window_sec = float(text)
        except ValueError:
            self.raw_window_sec = DEFAULT_RAW_WINDOW_SEC
        self._configure_slider()
        self._set_raw_start(self.raw_start, redraw=True)

    def on_amp_scale_changed(self, text):
        try:
            self.raw_amp_scale = float(text.rstrip("x"))
        except ValueError:
            self.raw_amp_scale = 1.0
        self._draw_raw_window(self.raw_start)

    def on_slider_changed(self, value):
        if self.slider_updating:
            return
        self._set_raw_start(float(value), redraw=True, update_slider=False)

    def _set_raw_start(self, start, redraw=True, update_slider=True):
        if self.raw is None:
            return
        max_start = max(0.0, self.total_sec - self.raw_window_sec)
        self.raw_start = max(0.0, min(max_start, float(start)))
        self.time_label.setText(sec2hms(self.raw_start))
        if update_slider:
            self.slider_updating = True
            self.time_slider.setValue(int(round(self.raw_start)))
            self.slider_updating = False
        if redraw:
            self._draw_raw_window(self.raw_start)

    def _move_raw_window(self, delta):
        self._set_raw_start(self.raw_start + float(delta), redraw=True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self._move_raw_window(-self.raw_window_sec)
        elif event.key() == Qt.Key_Right:
            self._move_raw_window(self.raw_window_sec)
        else:
            super().keyPressEvent(event)

    def _draw_tfr(self):
        self.tfr_ax.clear()
        n_time = len(self.full_time)
        n_freq = len(self.freq_axis)
        ds_t = max(1, n_time // 1800)
        ds_f = max(1, n_freq // 450)
        t_disp = self.full_time[::ds_t]
        f_disp = self.freq_axis[::ds_f]
        spec_disp = self.spec_db[::ds_t, ::ds_f].astype(np.float32)

        self.tfr_ax.pcolormesh(
            t_disp, f_disp, spec_disp.T,
            shading="nearest", cmap="jet", vmin=V_MIN, vmax=V_MAX, rasterized=True,
        )
        self.tfr_ax.set_ylim(*FREQ_RANGE)
        self.tfr_ax.set_xlim(0, self.total_sec)
        self.tfr_ax.set_ylabel("Hz")
        self.tfr_ax.set_xlabel("Recording Time")
        self.tfr_ax.xaxis.set_major_formatter(FuncFormatter(sec2hm))
        self.tfr_ax.set_title(f"Whole-night TFR: drag the right edge of the white box; right-click to save W/NREM/REM | {os.path.basename(self.current_edf)}")
        self._ensure_pending_sleep_box()
        self._redraw_sleep_patches()
        self._redraw_pending_sleep_patch()
        self._connect_tfr_events()
        self.tfr_canvas.draw_idle()

    def _connect_tfr_events(self):
        if self.tfr_click_cid is not None:
            try:
                self.tfr_canvas.mpl_disconnect(self.tfr_click_cid)
            except Exception:
                pass
        if self.tfr_motion_cid is not None:
            try:
                self.tfr_canvas.mpl_disconnect(self.tfr_motion_cid)
            except Exception:
                pass
        if self.tfr_release_cid is not None:
            try:
                self.tfr_canvas.mpl_disconnect(self.tfr_release_cid)
            except Exception:
                pass
        self.tfr_click_cid = self.tfr_canvas.mpl_connect("button_press_event", self._on_tfr_click)
        self.tfr_motion_cid = self.tfr_canvas.mpl_connect("motion_notify_event", self._on_tfr_motion)
        self.tfr_release_cid = self.tfr_canvas.mpl_connect("button_release_event", self._on_tfr_release)

    def _on_tfr_click(self, event):
        if event.xdata is None or event.inaxes != self.tfr_ax:
            return
        if event.button == 1:
            if self._is_near_pending_sleep_edge(event.xdata):
                self.sleep_dragging = True
                self._set_pending_sleep_end(event.xdata, final=False)
                self.set_status("Dragging the right edge of the sleep-stage box")
            else:
                self.set_status("Hold and drag the right edge of the white dashed box")
            return
        if event.button != 3:
            return
        clicked_idx = None
        for idx, rec in enumerate(self.sleep_records):
            if rec["start"] <= event.xdata <= rec["end"]:
                clicked_idx = idx
                break

        if clicked_idx is not None:
            rec = self.sleep_records[clicked_idx]
            self.selected_sleep_idx = clicked_idx
            self.current_sleep_selection = (rec["start"], rec["end"])
            self.notes_edit.setText(rec.get("notes", ""))

        menu = QMenu(self)
        stage_menu = menu.addMenu("Save Current Box As")
        stage_actions = {}
        for stage in SLEEP_STAGES:
            action = stage_menu.addAction(stage)
            action.setEnabled(self.current_sleep_selection is not None)
            stage_actions[action] = stage

        menu.addSeparator()
        act_delete = menu.addAction("Delete Selected Stage")
        act_delete.setEnabled(clicked_idx is not None)

        pos = event.guiEvent.globalPos() if event.guiEvent is not None else self.tfr_canvas.mapToGlobal(self.tfr_canvas.rect().center())
        chosen = menu.exec_(pos)
        if chosen in stage_actions:
            self._save_sleep_stage(stage_actions[chosen])
        elif chosen == act_delete:
            self._delete_sleep_stage(clicked_idx)

    def _next_sleep_start(self):
        if self.sleep_records:
            return max(rec["end"] for rec in self.sleep_records)
        return 0.0

    def _ensure_pending_sleep_box(self):
        if self.selected_sleep_idx is not None or self.current_sleep_selection is not None:
            return
        start = self._next_sleep_start()
        if start >= self.total_sec:
            return
        end = min(self.total_sec, start + DEFAULT_SLEEP_BOX_SEC)
        self.current_sleep_selection = (start, end)

    def _is_near_pending_sleep_edge(self, x):
        if not self.current_sleep_selection:
            return False
        _, end = self.current_sleep_selection
        tolerance = max(SLEEP_EDGE_TOLERANCE_SEC, self.total_sec * 0.003)
        return abs(float(x) - end) <= tolerance

    def _set_pending_sleep_end(self, x, final=False):
        start = self._next_sleep_start()
        end = max(start, min(self.total_sec, float(x)))
        if end - start < 0.2:
            if final:
                self.set_status(f"Current sleep-stage box is too short: start {sec2hms(start)}. Drag the end edge to the right.")
            return
        self.current_sleep_selection = (start, end)
        self.selected_sleep_idx = None
        self._redraw_pending_sleep_patch()
        self.tfr_canvas.draw_idle()
        self.set_status(f"Current sleep-stage box: {sec2hms(start)} -> {sec2hms(end)}; right-click to save as W/NREM/REM")

    def _on_tfr_motion(self, event):
        if not self.sleep_dragging or event.xdata is None or event.inaxes != self.tfr_ax:
            return
        self._set_pending_sleep_end(event.xdata, final=False)

    def _on_tfr_release(self, event):
        if event.button != 1 or not self.sleep_dragging:
            return
        self.sleep_dragging = False
        if event.xdata is not None and event.inaxes == self.tfr_ax:
            self._set_pending_sleep_end(event.xdata, final=True)

    def _redraw_sleep_patches(self):
        for patch, text in self.sleep_patches:
            try:
                patch.remove()
                text.remove()
            except Exception:
                pass
        self.sleep_patches.clear()

        y0, y1 = FREQ_RANGE
        for rec in self.sleep_records:
            color = STAGE_COLORS.get(rec["stage"], "#607D8B")
            patch = Rectangle((rec["start"], y0), rec["end"] - rec["start"], y1 - y0,
                              facecolor=color, alpha=0.20, edgecolor=color, linewidth=1.2)
            self.tfr_ax.add_patch(patch)
            text = self.tfr_ax.text((rec["start"] + rec["end"]) / 2, y1 - 4, rec["stage"],
                                    ha="center", va="top", fontsize=9, color="white",
                                    bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=2))
            self.sleep_patches.append((patch, text))

    def _redraw_pending_sleep_patch(self):
        for artist_name in ("pending_sleep_patch", "pending_sleep_edge", "pending_sleep_text"):
            artist = getattr(self, artist_name, None)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(self, artist_name, None)
        if not self.current_sleep_selection or self.selected_sleep_idx is not None:
            return
        start, end = self.current_sleep_selection
        if end <= start:
            return
        y0, y1 = FREQ_RANGE
        self.pending_sleep_patch = Rectangle(
            (start, y0), end - start, y1 - y0,
            facecolor="#FFFFFF", alpha=0.12, edgecolor="#FFFFFF",
            linewidth=2.0, linestyle="--",
        )
        self.tfr_ax.add_patch(self.pending_sleep_patch)
        self.pending_sleep_edge = self.tfr_ax.axvline(end, color="#FFFFFF", lw=3.0, alpha=0.95)
        self.pending_sleep_text = self.tfr_ax.text(
            end, y1 - 3, "Drag edge",
            ha="center", va="top", fontsize=9, color="#111827",
            bbox=dict(facecolor="#FFFFFF", alpha=0.9, edgecolor="none", pad=2),
        )

    def _draw_raw_window(self, start):
        if self.raw is None:
            return
        start = max(0.0, float(start))
        end = min(self.total_sec, start + self.raw_window_sec)
        if end <= start:
            end = min(self.total_sec, start + DEFAULT_RAW_WINDOW_SEC)

        self.raw_ax.clear()
        try:
            seg = self.raw.copy().crop(tmin=start, tmax=end, include_tmax=True)
            seg.load_data()
            data = seg.get_data(picks=self.raw_channels) * 1e6
            times = seg.times + start
        except Exception as exc:
            self.raw_ax.text(0.5, 0.5, f"Failed to read raw data\n{exc}", transform=self.raw_ax.transAxes,
                             ha="center", va="center")
            self.raw_canvas.draw_idle()
            return

        if data.shape[1] > MAX_RAW_POINTS:
            step = int(np.ceil(data.shape[1] / MAX_RAW_POINTS))
            data = data[:, ::step]
            times = times[::step]

        scale = np.nanpercentile(np.abs(data), 95)
        if not np.isfinite(scale) or scale <= 0:
            scale = 50.0
        offset_step = scale * 3.0
        offsets = np.arange(data.shape[0])[::-1] * offset_step

        for i, ch in enumerate(self.raw_channels):
            self.raw_ax.plot(times, data[i] * self.raw_amp_scale + offsets[i], lw=0.7, color="#263238")

        for idx, rec in enumerate(self.seizure_records):
            rec_start = rec["start"]
            rec_end = rec["end"]
            if rec_end >= start and rec_start <= end:
                selected = idx == self.selected_seizure_idx
                span_start = max(start, rec_start)
                span_end = min(end, rec_end)
                color = "#7B3294" if selected else "#E69F00"
                self.raw_ax.axvspan(span_start, span_end, color=color, alpha=0.22 if selected else 0.12)
                self.raw_ax.axvline(rec_start, color=color, lw=2.2 if selected else 1.5, alpha=0.95)
                self.raw_ax.axvline(rec_end, color=color, lw=2.2 if selected else 1.5, alpha=0.95)
                label_x = min(max((rec_start + rec_end) / 2, start), end)
                self.raw_ax.text(label_x, offsets[0] + offset_step * 0.35, rec.get("seizure_type", "Seizure"),
                                 color="#E69F00", ha="center", va="bottom", fontsize=9)

        if self.current_seizure_start is not None and start <= self.current_seizure_start <= end:
            self.raw_ax.axvline(self.current_seizure_start, color="#7B3294", lw=1.8, linestyle="--")
            self.raw_ax.text(self.current_seizure_start, offsets[0] + offset_step * 0.65, "Start",
                             color="#7B3294", ha="center", va="bottom", fontsize=9)
        if self.current_seizure_end is not None and start <= self.current_seizure_end <= end:
            self.raw_ax.axvline(self.current_seizure_end, color="#E69F00", lw=1.8, linestyle="--")
            self.raw_ax.text(self.current_seizure_end, offsets[0] + offset_step * 0.65, "End",
                             color="#E69F00", ha="center", va="bottom", fontsize=9)

        self.raw_ax.set_yticks(offsets)
        self.raw_ax.set_yticklabels(self.raw_channels, fontsize=8)
        self.raw_ax.set_xlim(start, end)
        self.raw_ax.set_xlabel("Recording Time")
        self.raw_ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: sec2hms(x)))
        self.raw_ax.set_title(
            f"Raw Waveforms, First 16 Channels: amplitude {self.raw_amp_scale:g}x | left-click saved seizures, right-click for actions | "
            f"{sec2hms(start)} -> {sec2hms(end)}"
        )
        self.raw_ax.grid(True, axis="x", alpha=0.25, linestyle="--")
        self._connect_raw_events()
        self.raw_canvas.draw_idle()

    def _connect_raw_events(self):
        if self.raw_click_cid is None:
            self.raw_click_cid = self.raw_canvas.mpl_connect("button_press_event", self._on_raw_click)
        if self.raw_scroll_cid is None:
            self.raw_scroll_cid = self.raw_canvas.mpl_connect("scroll_event", self._on_raw_scroll)

    def _on_raw_click(self, event):
        if self.raw is None or event.inaxes != self.raw_ax or event.xdata is None:
            return
        t = max(0.0, min(self.total_sec, float(event.xdata)))
        if event.button == 3:
            self._show_raw_context_menu(event, t)
            return
        if event.button != 1:
            return

        clicked_idx = self._find_seizure_at_time(t)
        if clicked_idx is not None:
            self._select_seizure(clicked_idx, jump=False)
            self._draw_raw_window(self.raw_start)
            rec = self.seizure_records[clicked_idx]
            self.set_status(f"Selected seizure #{clicked_idx+1}: {sec2hms(rec['start'])} -> {sec2hms(rec['end'])}")
            return
        self.selected_seizure_idx = None
        self._update_seizure_buttons()
        self._draw_raw_window(self.raw_start)

    def _on_raw_scroll(self, event):
        if self.raw is None:
            return
        delta = -self.raw_window_sec if event.step > 0 else self.raw_window_sec
        self._move_raw_window(delta)

    def _show_raw_context_menu(self, event, t):
        clicked_idx = self._find_seizure_at_time(t)
        if clicked_idx is not None:
            self._select_seizure(clicked_idx, jump=False)

        menu = QMenu(self)
        scope_menu = menu.addMenu(f"Scope: {self.current_scope}")
        scope_actions = {}
        for scope in SCOPES:
            action = scope_menu.addAction(scope)
            action.setCheckable(True)
            action.setChecked(scope == self.current_scope)
            scope_actions[action] = scope

        type_menu = menu.addMenu(f"Seizure Type: {self.current_seizure_type}")
        type_actions = {}
        for seizure_type in SEIZURE_TYPES:
            action = type_menu.addAction(seizure_type)
            action.setCheckable(True)
            action.setChecked(seizure_type == self.current_seizure_type)
            type_actions[action] = seizure_type

        menu.addSeparator()
        act_start = menu.addAction(f"Set Seizure Start: {sec2hms(t)}")
        act_end = menu.addAction(f"Set Seizure End: {sec2hms(t)}")
        menu.addSeparator()
        act_delete = menu.addAction("Delete Selected Seizure")
        act_delete.setEnabled(self.selected_seizure_idx is not None)
        act_clear = menu.addAction("Clear Current Start/End")
        act_clear.setEnabled(self.current_seizure_start is not None or self.current_seizure_end is not None)

        pos = event.guiEvent.globalPos() if event.guiEvent is not None else self.raw_canvas.mapToGlobal(self.raw_canvas.rect().center())
        chosen = menu.exec_(pos)
        if chosen in scope_actions:
            self.current_scope = scope_actions[chosen]
            self.set_status(f"Current seizure scope: {self.current_scope}")
        elif chosen in type_actions:
            self.current_seizure_type = type_actions[chosen]
            self.set_status(f"Current seizure type: {self.current_seizure_type}")
        elif chosen == act_start:
            self._set_current_seizure_start(t)
        elif chosen == act_end:
            self._set_current_seizure_end(t)
        elif chosen == act_delete:
            self.on_delete_selected_seizure()
        elif chosen == act_clear:
            self.on_clear_current_seizure()

    def _set_current_seizure_start(self, t):
        self.current_seizure_start = round(float(t), 2)
        if self.current_seizure_end is not None and self.current_seizure_end <= self.current_seizure_start:
            self.current_seizure_end = None
        elif self.current_seizure_end is not None:
            self._save_current_seizure(auto=True)
            return
        self._update_seizure_buttons()
        self._draw_raw_window(self.raw_start)
        self.set_status(f"Seizure start set: {sec2hms(t)}")

    def _set_current_seizure_end(self, t):
        self.current_seizure_end = round(float(t), 2)
        if self.current_seizure_start is not None and self.current_seizure_end <= self.current_seizure_start:
            QMessageBox.warning(self, "Time Error", "Seizure end time must be later than start time.")
            self.current_seizure_end = None
            self._draw_raw_window(self.raw_start)
            return
        if self.current_seizure_start is not None:
            self._save_current_seizure(auto=True)
            return
        self._update_seizure_buttons()
        self._draw_raw_window(self.raw_start)
        self.set_status(f"Seizure end set: {sec2hms(t)}")

    def on_mark_start_mode(self):
        self.raw_mark_mode = "start"
        self._draw_raw_window(self.raw_start)
        self.set_status("Current mode: click the raw waveform below to set seizure start time")

    def on_mark_end_mode(self):
        self.raw_mark_mode = "end"
        self._draw_raw_window(self.raw_start)
        self.set_status("Current mode: click the raw waveform below to set seizure end time")

    def on_clear_current_seizure(self):
        self.current_seizure_start = None
        self.current_seizure_end = None
        self.selected_seizure_idx = None
        self._update_seizure_buttons()
        self._draw_raw_window(self.raw_start)
        self.set_status("Cleared the current unsaved seizure start/end times")

    def _update_seizure_buttons(self):
        has_records = bool(self.seizure_records)
        self.seizure_combo.setEnabled(has_records)
        self.btn_jump_seizure.setEnabled(has_records)

    def on_save_seizure(self):
        self._save_current_seizure(auto=False)

    def _save_current_seizure(self, auto=False):
        if self.current_seizure_start is None or self.current_seizure_end is None:
            QMessageBox.information(self, "Tip", "Please mark both seizure start and end times in the raw waveform below.")
            return
        if self.current_seizure_end <= self.current_seizure_start:
            QMessageBox.warning(self, "Time Error", "Seizure end time must be later than start time.")
            return

        rec = {
            "start": round(self.current_seizure_start, 2),
            "end": round(self.current_seizure_end, 2),
            "scope": self.current_scope,
            "seizure_type": self.current_seizure_type,
            "notes": self.notes_edit.text().strip(),
        }
        self.seizure_records.append(rec)
        self.seizure_records.sort(key=lambda r: r["start"])
        self.selected_seizure_idx = self.seizure_records.index(rec)
        self._write_seizure_times()
        self.current_seizure_start = None
        self.current_seizure_end = None
        self._refresh_seizure_combo()
        self._update_seizure_buttons()
        self._draw_raw_window(self.raw_start)
        self.set_status(
            f"{'Auto-saved' if auto else 'Saved'} seizure: {sec2hms(rec['start'])} -> {sec2hms(rec['end'])} | "
            f"{rec['scope']} | {rec['seizure_type']}"
        )

    def _refresh_seizure_combo(self):
        current_idx = self.selected_seizure_idx
        self.seizure_combo.blockSignals(True)
        self.seizure_combo.clear()
        for idx, rec in enumerate(self.seizure_records):
            self.seizure_combo.addItem(
                f"#{idx+1} {sec2hms(rec['start'])}-{sec2hms(rec['end'])} {rec.get('seizure_type', '')}",
                idx,
            )
        if current_idx is not None and 0 <= current_idx < len(self.seizure_records):
            self.seizure_combo.setCurrentIndex(current_idx)
        self.seizure_combo.blockSignals(False)
        self._update_seizure_buttons()

    def _find_seizure_at_time(self, t):
        tolerance = max(0.2, self.raw_window_sec * 0.01)
        for idx, rec in enumerate(self.seizure_records):
            if rec["start"] - tolerance <= t <= rec["end"] + tolerance:
                return idx
        return None

    def _select_seizure(self, idx, jump=True):
        if idx is None or idx < 0 or idx >= len(self.seizure_records):
            self.selected_seizure_idx = None
            self._update_seizure_buttons()
            return
        self.selected_seizure_idx = idx
        self.seizure_combo.blockSignals(True)
        self.seizure_combo.setCurrentIndex(idx)
        self.seizure_combo.blockSignals(False)
        rec = self.seizure_records[idx]
        self.current_scope = rec.get("scope", "Uncertain")
        self.current_seizure_type = rec.get("seizure_type", "Suspected/uncertain seizure")
        self.notes_edit.setText(rec.get("notes", ""))
        if jump:
            target = rec["start"] - max(0.0, (self.raw_window_sec - (rec["end"] - rec["start"])) / 2)
            self._set_raw_start(target, redraw=True)
        self._update_seizure_buttons()

    def on_jump_selected_seizure(self):
        idx = self.seizure_combo.currentData()
        if idx is None:
            return
        self._select_seizure(int(idx), jump=True)
        rec = self.seizure_records[int(idx)]
        self.set_status(f"Jumped to seizure #{int(idx)+1}: {sec2hms(rec['start'])} -> {sec2hms(rec['end'])}")

    def on_seizure_combo_changed(self, _idx):
        idx = self.seizure_combo.currentData()
        if idx is None:
            self.selected_seizure_idx = None
        else:
            self._select_seizure(int(idx), jump=False)
        self._draw_raw_window(self.raw_start)

    def on_delete_selected_seizure(self):
        idx = self.selected_seizure_idx
        if idx is None:
            idx = self.seizure_combo.currentData()
        if idx is None or int(idx) < 0 or int(idx) >= len(self.seizure_records):
            QMessageBox.information(self, "Tip", "Please click a saved seizure in the raw waveform, or select one in the seizure table.")
            return
        idx = int(idx)
        rec = self.seizure_records.pop(idx)
        self.selected_seizure_idx = None
        self._write_seizure_times()
        self._refresh_seizure_combo()
        self._draw_raw_window(self.raw_start)
        self.set_status(f"Deleted seizure: {sec2hms(rec['start'])} -> {sec2hms(rec['end'])}")

    def _save_sleep_stage(self, stage):
        if not self.current_sleep_selection:
            QMessageBox.information(self, "Tip", "Please drag the right edge of the default box in the whole-night TFR first.")
            return
        selected_start, selected_end = self.current_sleep_selection
        start = selected_start
        end = selected_end
        if self.selected_sleep_idx is None and self.sleep_records:
            start = max(rec["end"] for rec in self.sleep_records)
        if end <= start:
            QMessageBox.warning(self, "Time Error", "End time must be later than the previous sleep-stage segment.")
            return

        rec = {
            "start": round(start, 2),
            "end": round(end, 2),
            "stage": stage,
            "notes": self.notes_edit.text().strip(),
        }
        if self.selected_sleep_idx is not None:
            self.sleep_records[self.selected_sleep_idx] = rec
            action = "Updated"
        else:
            self.sleep_records.append(rec)
            action = "Saved"
        self.sleep_records.sort(key=lambda r: r["start"])
        self.selected_sleep_idx = None
        self.current_sleep_selection = None
        self.pending_sleep_patch = None
        self.pending_sleep_edge = None
        self.pending_sleep_text = None
        self._write_sleep_stages()
        self._draw_tfr()
        self.set_status(f"{action} sleep stage: {sec2hms(start)} -> {sec2hms(end)} | {rec['stage']}")

    def _delete_sleep_stage(self, idx=None):
        if idx is None:
            idx = self.selected_sleep_idx
        if idx is None:
            QMessageBox.information(self, "Tip", "Please right-click an existing sleep-stage block first.")
            return
        rec = self.sleep_records.pop(idx)
        self.selected_sleep_idx = None
        self._write_sleep_stages()
        self._draw_tfr()
        self.set_status(f"Deleted sleep stage: {sec2hms(rec['start'])} -> {sec2hms(rec['end'])}")

    def on_undo_seizure(self):
        if not self.seizure_records:
            return
        rec = self.seizure_records.pop()
        self.selected_seizure_idx = None
        self._write_seizure_times()
        self._refresh_seizure_combo()
        self._draw_raw_window(self.raw_start)
        self.set_status(f"Undid seizure: {sec2hms(rec['start'])} -> {sec2hms(rec['end'])}")

    def _sleep_csv_path(self):
        if not self.current_edf:
            return None
        return SLEEP_CSV_DIR / f"{Path(self.current_edf).stem}_sleep_stages.csv"

    def _seizure_csv_path(self):
        if not self.current_edf:
            return None
        return SEIZURE_CSV_DIR / f"{Path(self.current_edf).stem}_seizure_times.csv"

    def _load_sleep_stages(self):
        self.sleep_records = []
        path = self._sleep_csv_path()
        if not path or not path.exists():
            return
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.sleep_records.append({
                    "start": float(row["start_sec"]),
                    "end": float(row["end_sec"]),
                    "stage": LEGACY_STAGE_MAP.get(row.get("stage", "NREM"), "NREM"),
                    "notes": row.get("notes", ""),
                })
        self.sleep_records.sort(key=lambda r: r["start"])

    def _load_seizure_times(self):
        self.seizure_records = []
        path = self._seizure_csv_path()
        if not path or not path.exists():
            return
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "start_sec" in row and row.get("start_sec"):
                    start = float(row["start_sec"])
                    end = float(row.get("end_sec") or start)
                else:
                    start = float(row["time_sec"])
                    end = start + 0.01
                self.seizure_records.append({
                    "start": start,
                    "end": end,
                    "scope": row.get("scope", "Uncertain"),
                    "seizure_type": row.get("seizure_type", "Suspected/uncertain seizure"),
                    "notes": row.get("notes", ""),
                })
        self.seizure_records.sort(key=lambda r: r["start"])

    def _write_sleep_stages(self):
        path = self._sleep_csv_path()
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "start_sec", "end_sec", "duration_sec", "stage", "notes"])
            for rec in sorted(self.sleep_records, key=lambda r: r["start"]):
                writer.writerow([
                    os.path.basename(self.current_edf),
                    f"{rec['start']:.2f}",
                    f"{rec['end']:.2f}",
                    f"{rec['end'] - rec['start']:.2f}",
                    rec["stage"],
                    rec.get("notes", ""),
                ])

    def _write_seizure_times(self):
        path = self._seizure_csv_path()
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "file", "start_sec", "end_sec", "duration_sec", "start_hms", "end_hms",
                "scope", "seizure_type", "notes",
            ])
            for rec in sorted(self.seizure_records, key=lambda r: r["start"]):
                writer.writerow([
                    os.path.basename(self.current_edf),
                    f"{rec['start']:.2f}",
                    f"{rec['end']:.2f}",
                    f"{rec['end'] - rec['start']:.2f}",
                    sec2hms(rec["start"]),
                    sec2hms(rec["end"]),
                    rec.get("scope", ""),
                    rec.get("seizure_type", ""),
                    rec.get("notes", ""),
                ])

    def set_status(self, msg):
        self.status_label.setText(msg)
        print(f"[Status] {msg}")

    def closeEvent(self, event):
        if self.raw is not None:
            try:
                self.raw.close()
            except Exception:
                pass
        plt.close("all")
        super().closeEvent(event)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 16))
    window = SeizureLabelGUI()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
