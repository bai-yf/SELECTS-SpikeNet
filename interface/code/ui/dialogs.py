
"""Dialogs and fast sleep-stage/TXT selection helpers."""
import concurrent.futures
import io
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from PyQt5.QtCore import QTime
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from config import MANUAL_SLEEP_LABEL_DIR, MANUAL_SLEEP_STAGE_GUI
from core.preprocessing import warm_up_matlab_engine_async
from utils.eeg_io import read_supported_eeg, resolve_eeg_file_path

class SpikeConfirmDialog(QDialog):
    def __init__(self, spike_time, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Spike")
        self.setFixedSize(420, 190)

        layout = QVBoxLayout()
        self.label = QLabel(f"Confirm the region at {spike_time:.3f} s as a spike?")
        self.label.setStyleSheet("font-size: 20px;")
        layout.addWidget(self.label)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Ok).setText("Confirm (Purple)")
        button_box.button(QDialogButtonBox.Cancel).setText("Reject (Remove)")
        layout.addWidget(button_box)

        self.setLayout(layout)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)


class TimeRowRangeDialog(QDialog):
    def __init__(self, total_rows, fs=500, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Data Range (Rows or Time)")
        self.setFixedSize(560, 380)
        self.total_rows = total_rows
        self.fs = fs
        self.total_seconds = total_rows / fs

        self.selected_start_row = 1
        self.selected_end_row = total_rows
        self.selected_start_time = QTime(0, 0, 0)
        self.selected_end_time = self.seconds_to_qtime(self.total_seconds)

        layout = QVBoxLayout()

        self.info_label = QLabel(
            f"File rows: {total_rows} (sampling rate: {fs} Hz, total duration: {self.format_seconds(self.total_seconds)})\n"
            "Select the range to keep:"
        )
        self.info_label.setStyleSheet("font-size: 20px;")
        layout.addWidget(self.info_label)

        row_group = QWidget()
        row_layout = QVBoxLayout(row_group)
        row_layout.addWidget(QLabel("--- Row Range ---"))

        start_row_layout = QHBoxLayout()
        start_row_layout.addWidget(QLabel("Start Row:"))
        self.start_row_input = QSpinBox()
        self.start_row_input.setRange(1, total_rows)
        self.start_row_input.setValue(1)
        start_row_layout.addWidget(self.start_row_input)
        row_layout.addLayout(start_row_layout)

        end_row_layout = QHBoxLayout()
        end_row_layout.addWidget(QLabel("End Row:"))
        self.end_row_input = QSpinBox()
        self.end_row_input.setRange(1, total_rows)
        self.end_row_input.setValue(total_rows)
        end_row_layout.addWidget(self.end_row_input)
        row_layout.addLayout(end_row_layout)
        layout.addWidget(row_group)

        time_group = QWidget()
        time_layout = QVBoxLayout(time_group)
        time_layout.addWidget(QLabel("--- Time Range (HH:MM:SS) ---"))

        start_time_layout = QHBoxLayout()
        start_time_layout.addWidget(QLabel("Start Time:"))
        self.start_time_input = QTimeEdit()
        self.start_time_input.setDisplayFormat("HH:mm:ss")
        self.start_time_input.setTime(self.selected_start_time)
        start_time_layout.addWidget(self.start_time_input)
        time_layout.addLayout(start_time_layout)

        end_time_layout = QHBoxLayout()
        end_time_layout.addWidget(QLabel("End Time:"))
        self.end_time_input = QTimeEdit()
        self.end_time_input.setDisplayFormat("HH:mm:ss")
        self.end_time_input.setTime(self.selected_end_time)
        end_time_layout.addWidget(self.end_time_input)
        time_layout.addLayout(end_time_layout)
        layout.addWidget(time_group)

        btn_layout = QHBoxLayout()
        self.all_btn = QPushButton("Keep All")
        self.all_btn.clicked.connect(self.select_all)
        btn_layout.addWidget(self.all_btn)
        layout.addLayout(btn_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Ok).setText("Confirm")
        button_box.button(QDialogButtonBox.Cancel).setText("Cancel")
        layout.addWidget(button_box)

        self.setLayout(layout)

        button_box.accepted.connect(self.on_accept)
        button_box.rejected.connect(self.reject)
        self.start_time_input.timeChanged.connect(self.update_start_row_from_time)
        self.end_time_input.timeChanged.connect(self.update_end_row_from_time)
        self.start_row_input.valueChanged.connect(self.update_start_time_from_row)
        self.end_row_input.valueChanged.connect(self.update_end_time_from_row)

    def qtime_to_seconds(self, qtime):
        return qtime.hour() * 3600 + qtime.minute() * 60 + qtime.second()

    def seconds_to_qtime(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return QTime(hours, minutes, secs)

    def format_seconds(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def update_start_row_from_time(self):
        secs = self.qtime_to_seconds(self.start_time_input.time())
        row = int(secs * self.fs) + 1
        row = max(1, min(row, self.total_rows))
        self.start_row_input.setValue(row)

    def update_end_row_from_time(self):
        secs = self.qtime_to_seconds(self.end_time_input.time())
        row = int(secs * self.fs) + 1
        row = max(1, min(row, self.total_rows))
        self.end_row_input.setValue(row)

    def update_start_time_from_row(self):
        row = self.start_row_input.value()
        secs = (row - 1) / self.fs
        self.start_time_input.setTime(self.seconds_to_qtime(secs))

    def update_end_time_from_row(self):
        row = self.end_row_input.value()
        secs = (row - 1) / self.fs
        self.end_time_input.setTime(self.seconds_to_qtime(secs))

    def select_all(self):
        self.start_row_input.setValue(1)
        self.end_row_input.setValue(self.total_rows)
        self.start_time_input.setTime(QTime(0, 0, 0))
        self.end_time_input.setTime(self.seconds_to_qtime(self.total_seconds))

    def on_accept(self):
        self.selected_start_row = self.start_row_input.value()
        self.selected_end_row = self.end_row_input.value()
        if self.selected_start_row > self.selected_end_row:
            QMessageBox.warning(self, "Input Error", "Start row cannot be greater than end row.")
            return
        self.accept()


def parse_sleep_stage_file(sleep_file_path):
    try:
        sleep_df = pd.read_csv(sleep_file_path)
        sleep_df.columns = [str(col).strip() for col in sleep_df.columns]
        lower_to_original = {col.lower(): col for col in sleep_df.columns}

        # Source 1: exported staging CSV with latency,type; each row defaults to 30 seconds.
        if 'latency' in lower_to_original and 'type' in lower_to_original:
            latency_col = lower_to_original['latency']
            stage_col = lower_to_original['type']
            sleep_df[latency_col] = pd.to_numeric(sleep_df[latency_col], errors='coerce')
            sleep_df = sleep_df.dropna(subset=[latency_col])
            stage_text = sleep_df[stage_col].astype(str).str.upper().str.strip()
            n_stages = sleep_df[stage_text.isin(['N1', 'N2', 'N3', 'NREM'])].copy()
            if len(n_stages) == 0:
                QMessageBox.warning(None, "No Sleep Stage Data", "No N1/N2/N3/NREM stages were found in the sleep-stage file.")
                return None, None, None
            n_stages = n_stages.sort_values(latency_col)
            sleep_intervals = [(float(t), float(t) + 30.0, str(stage).upper().strip())
                               for t, stage in zip(n_stages[latency_col], n_stages[stage_col])]
            return sleep_intervals[0][0], sleep_intervals[-1][1], sleep_intervals

        # Source 2: manual GUI labels with start_sec,end_sec,stage.
        required = {'start_sec', 'end_sec', 'stage'}
        if required.issubset(lower_to_original):
            start_col = lower_to_original['start_sec']
            end_col = lower_to_original['end_sec']
            stage_col = lower_to_original['stage']
            sleep_df[start_col] = pd.to_numeric(sleep_df[start_col], errors='coerce')
            sleep_df[end_col] = pd.to_numeric(sleep_df[end_col], errors='coerce')
            sleep_df = sleep_df.dropna(subset=[start_col, end_col])
            stage_text = sleep_df[stage_col].astype(str).str.upper().str.strip()
            n_stages = sleep_df[stage_text.isin(['N1', 'N2', 'N3', 'NREM'])].copy()
            if len(n_stages) == 0:
                QMessageBox.warning(None, "No Sleep Stage Data", "No NREM/N1/N2/N3 stages were found in the manual annotation file.")
                return None, None, None
            n_stages = n_stages.sort_values(start_col)
            sleep_intervals = []
            for _, row in n_stages.iterrows():
                start = float(row[start_col])
                end = float(row[end_col])
                if end > start:
                    sleep_intervals.append((start, end, str(row[stage_col]).upper().strip()))
            if not sleep_intervals:
                QMessageBox.warning(None, "No Valid Sleep Stage", "The NREM segment start/end times in the manual annotation file are invalid.")
                return None, None, None
            return sleep_intervals[0][0], sleep_intervals[-1][1], sleep_intervals

        raise ValueError("CSV must contain latency,type or manual annotation fields start_sec,end_sec,stage")

    except Exception as e:
        QMessageBox.critical(None, "Sleep-Stage File Parse Failed", "Error: " + str(e) + "\nSupported formats: 1) latency,type; 2) start_sec,end_sec,stage")
        return None, None, None


def extract_base_offset_from_filename(sleep_filename):
    # 睡眠分期CSV中的latency已经是以秒为单位、从数据开始计时；不再从文件名提取偏移。
    return 0


def calculate_txt_row_range(base_offset, first_n_latency, last_n_latency, fs=500):
    start_row = int((base_offset + first_n_latency) * fs) + 1
    end_row = int((base_offset + last_n_latency) * fs)
    return start_row, end_row


def detect_txt_header_rows(file_path):
    """Return 1 when the first non-empty TXT row is a channel-name header."""
    with open(file_path, 'rb') as file_obj:
        first_line = file_obj.readline().decode('utf-8', errors='ignore').strip()
    if not first_line:
        return 0
    first_value = first_line.replace(',', ' ').replace('\t', ' ').split()[0]
    try:
        float(first_value)
        return 0
    except ValueError:
        return 1


def build_txt_sparse_index(file_path, required_physical_row, stride=100000):
    """Build in-memory byte offsets for every `stride` physical TXT rows."""
    offsets = [0]
    newline_count = 0
    absolute_offset = 0
    chunk_size = 64 * 1024 * 1024
    with open(file_path, 'rb', buffering=chunk_size) as file_obj:
        while newline_count < required_physical_row:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            newline_positions = np.flatnonzero(np.frombuffer(chunk, dtype=np.uint8) == 10)
            if newline_positions.size:
                ordinals = newline_count + np.arange(1, newline_positions.size + 1, dtype=np.int64)
                selected = newline_positions[(ordinals % stride) == 0]
                offsets.extend((absolute_offset + selected + 1).tolist())
                newline_count += int(newline_positions.size)
            absolute_offset += len(chunk)

    indexed_rows = newline_count + 1
    if indexed_rows < required_physical_row:
        raise ValueError(
            f"TXT ends at approximately row {indexed_rows:,}, but row {required_physical_row:,} is required. "
            "Please confirm that the sleep-stage CSV matches this TXT file."
        )

    return np.asarray(offsets, dtype=np.int64)


def read_txt_intervals_fast(file_path, interval_rows, channel_count=16):
    """Scan the TXT once and parse only the requested 1-based row intervals."""
    if not interval_rows:
        raise ValueError("No TXT intervals were provided.")

    header_rows = detect_txt_header_rows(file_path)
    physical_intervals = [
        (int(start) + header_rows, int(end) + header_rows)
        for start, end, _ in interval_rows
    ]
    expected_rows = sum(end - start + 1 for start, end in physical_intervals)
    output = np.empty((channel_count, expected_rows), dtype=np.float32)
    valid_mask = np.ones(expected_rows, dtype=bool)
    interval_offsets = []
    output_offset = 0
    for start_row, end_row in physical_intervals:
        interval_offsets.append(output_offset)
        output_offset += end_row - start_row + 1

    def parse_block(block, output_start, row_count):
        frame = pd.read_csv(
                io.BytesIO(block),
                sep='\t',
                header=None,
                usecols=list(range(channel_count)),
                dtype=np.float32,
                engine='c',
                skip_blank_lines=False,
            )
        values = frame.to_numpy(dtype=np.float32, copy=False)
        if len(values) != row_count:
            raise ValueError(
                f"Expected {row_count:,} TXT rows in a selected block, but read {len(values):,}."
            )
        block_valid = np.isfinite(values).all(axis=1)
        output[:, output_start:output_start + row_count] = values.T
        valid_mask[output_start:output_start + row_count] = block_valid
        return int((~block_valid).sum())

    # Parse selected byte blocks while the file is scanned sequentially. This
    # avoids the old second disk pass over all selected intervals. A bounded
    # queue prevents large byte blocks from accumulating in memory.
    pending = set()
    invalid_count = 0
    carry = b""
    first_physical_row = 1
    interval_index = 0
    chunk_size = 64 * 1024 * 1024
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        with open(file_path, 'rb', buffering=chunk_size) as file_obj:
            while interval_index < len(physical_intervals):
                chunk = file_obj.read(chunk_size)
                if chunk:
                    data = carry + chunk
                elif carry:
                    data = carry + b'\n'
                    carry = b""
                else:
                    break

                last_newline = data.rfind(b'\n')
                if last_newline < 0:
                    carry = data
                    continue
                complete = data[:last_newline + 1]
                carry = data[last_newline + 1:]
                newline_positions = np.flatnonzero(np.frombuffer(complete, dtype=np.uint8) == 10)
                line_count = int(newline_positions.size)
                last_physical_row = first_physical_row + line_count - 1

                while interval_index < len(physical_intervals):
                    start_row, end_row = physical_intervals[interval_index]
                    if start_row > last_physical_row:
                        break
                    overlap_start = max(start_row, first_physical_row)
                    overlap_end = min(end_row, last_physical_row)
                    if overlap_end >= overlap_start:
                        local_start = overlap_start - first_physical_row
                        local_end = overlap_end - first_physical_row
                        byte_start = 0 if local_start == 0 else int(newline_positions[local_start - 1]) + 1
                        byte_end = int(newline_positions[local_end]) + 1
                        block_output_start = interval_offsets[interval_index] + overlap_start - start_row
                        pending.add(executor.submit(
                            parse_block,
                            complete[byte_start:byte_end],
                            block_output_start,
                            overlap_end - overlap_start + 1,
                        ))
                        if len(pending) >= 8:
                            completed, pending = concurrent.futures.wait(
                                pending,
                                return_when=concurrent.futures.FIRST_COMPLETED,
                            )
                            invalid_count += sum(job.result() for job in completed)
                    if end_row <= last_physical_row:
                        interval_index += 1
                    else:
                        break
                first_physical_row += line_count
        invalid_count += sum(job.result() for job in pending)

    if interval_index < len(physical_intervals):
        required_row = physical_intervals[interval_index][1]
        raise ValueError(
            f"TXT ends at approximately row {first_physical_row - 1:,}, but row {required_row:,} is required. "
            "Please confirm that the sleep-stage CSV matches this TXT file."
        )
    if invalid_count:
        output = output[:, valid_mask]
    return output, invalid_count, header_rows


class SleepStageImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Data by Sleep Stage")
        self.setFixedSize(760, 560)

        self.txt_file_path = None
        self.data_file_path = None
        self.selected_start_time = None
        self.selected_end_time = None
        self.selected_start_row = None
        self.selected_end_row = None
        self.total_rows = None
        self.fs = 500
        self.raw_header = None

        layout = QVBoxLayout()

        self.sleep_source_label = QLabel("Sleep-stage source: choose an existing CSV, or open the manual staging tool to create one.")
        layout.addWidget(self.sleep_source_label)

        source_btn_row = QHBoxLayout()
        self.sleep_file_label = QLabel("No sleep-stage file selected")
        self.sleep_file_btn = QPushButton("1A. Select Sleep-Stage CSV")
        self.sleep_file_btn.clicked.connect(self.select_sleep_file)
        source_btn_row.addWidget(self.sleep_file_btn)

        self.manual_sleep_btn = QPushButton("1B. Manual Sleep Staging")
        self.manual_sleep_btn.clicked.connect(self.open_manual_sleep_label_gui)
        source_btn_row.addWidget(self.manual_sleep_btn)
        layout.addLayout(source_btn_row)
        layout.addWidget(self.sleep_file_label)

        self.txt_file_label = QLabel("No TXT/EDF/SET file selected")
        self.txt_file_btn = QPushButton("2. Select Matching EEG TXT/EDF/SET File")
        self.txt_file_btn.clicked.connect(self.select_txt_file)
        self.txt_file_btn.setEnabled(False)
        layout.addWidget(self.txt_file_btn)
        layout.addWidget(self.txt_file_label)

        self.result_group = QWidget()
        self.result_layout = QVBoxLayout(self.result_group)
        self.result_label = QLabel("Waiting for file selection...")
        self.result_layout.addWidget(self.result_label)
        layout.addWidget(self.result_group)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Ok).setText("Import This Range")
        button_box.button(QDialogButtonBox.Cancel).setText("Cancel")
        button_box.accepted.connect(self.on_accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        self.ok_button = button_box.button(QDialogButtonBox.Ok)
        layout.addWidget(button_box)

        self.setLayout(layout)

        self.sleep_file_path = None
        self.first_n_latency = None
        self.last_n_latency = None
        self.sleep_intervals = []
        self.base_offset = None
        self.record_end_time = None

    def open_manual_sleep_label_gui(self):
        os.makedirs(MANUAL_SLEEP_LABEL_DIR, exist_ok=True)
        if not os.path.exists(MANUAL_SLEEP_STAGE_GUI):
            QMessageBox.critical(self, "Manual Tool Missing", "Not found:\n" + MANUAL_SLEEP_STAGE_GUI)
            return
        env = os.environ.copy()
        env["MANUAL_SLEEP_LABEL_DIR"] = MANUAL_SLEEP_LABEL_DIR
        try:
            subprocess.Popen([sys.executable, MANUAL_SLEEP_STAGE_GUI], cwd=os.getcwd(), env=env)
        except Exception as e:
            QMessageBox.critical(self, "Manual Tool Launch Failed", str(e))
            return
        QMessageBox.information(
            self,
            "Manual Tool Opened",
            "Select an EDF/SET file in the manual staging tool and save sleep stages.\n"
            + "Labels will be saved to:\n" + MANUAL_SLEEP_LABEL_DIR + "\n\n"
            + "After saving, return here and click 1A. Select Sleep-Stage CSV to choose the generated *_sleep_stages.csv file."
        )

    def select_sleep_file(self):
        default_dir = MANUAL_SLEEP_LABEL_DIR if os.path.isdir(MANUAL_SLEEP_LABEL_DIR) else os.getcwd()
        sleep_file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Sleep-Stage File", default_dir, "CSV files (*.csv);;All files (*.*)"
        )
        if not sleep_file_path:
            return

        self.sleep_file_path = sleep_file_path
        sleep_filename = os.path.basename(sleep_file_path)
        self.sleep_file_label.setText(f"Sleep-stage file:\n{sleep_filename}")

        self.first_n_latency, self.last_n_latency, self.sleep_intervals = parse_sleep_stage_file(sleep_file_path)
        if self.first_n_latency is None or self.last_n_latency is None or not self.sleep_intervals:
            self.txt_file_btn.setEnabled(False)
            return

        # Start MATLAB as soon as a valid staging file is available.  File
        # selection plus the one-time TXT import usually hides the cold start.
        warm_up_matlab_engine_async()

        # Estimate the full recording duration from the staging CSV without
        # scanning a multi-gigabyte TXT merely to count its rows.
        try:
            stage_frame = pd.read_csv(sleep_file_path)
            stage_frame.columns = [str(col).strip().lower() for col in stage_frame.columns]
            if 'end_sec' in stage_frame.columns:
                self.record_end_time = float(pd.to_numeric(stage_frame['end_sec'], errors='coerce').max())
            elif 'latency' in stage_frame.columns:
                self.record_end_time = float(pd.to_numeric(stage_frame['latency'], errors='coerce').max()) + 30.0
        except Exception:
            self.record_end_time = None

        self.base_offset = extract_base_offset_from_filename(sleep_filename)
        if self.base_offset is None:
            self.base_offset = 0
            QMessageBox.information(self, "Base Offset", "Sleep-stage latency is calculated with a 0-second offset.")

        self.txt_file_btn.setEnabled(True)
        self.result_label.setText(
            f"Sleep-stage parse result:\n"
            f"1. First N-stage latency: {self.first_n_latency} s\n"
            f"2. Last N-stage latency: {self.last_n_latency} s\n"
            f"3. N-stage segments: {len(self.sleep_intervals)}\n"
            f"4. Base offset: {self.base_offset}\n"
            f"Select the matching TXT/EDF/SET file to calculate the data range."
        )

    def select_txt_file(self):
        data_file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Matching EEG TXT/EDF/SET File",
            r"E:\BECT(Rolandic)\Selects\display" if os.path.isdir(r"E:\BECT(Rolandic)\Selects\display") else os.getcwd(),
            "EEG files (*.txt *.edf *.set *.fdt);;TXT files (*.txt);;EDF files (*.edf);;SET files (*.set);;FDT files (*.fdt);;All files (*.*)"
        )
        if not data_file_path:
            return

        self.txt_file_path = data_file_path
        self.data_file_path = data_file_path
        data_filename = os.path.basename(data_file_path)
        data_ext = os.path.splitext(data_file_path)[1].lower()
        self.raw_header = None
        self.txt_file_label.setText(f"EEG file:\n{data_filename}")

        self.selected_start_time = float(self.first_n_latency)
        self.selected_end_time = float(self.last_n_latency) + 30.0

        if data_ext == ".txt":
            warm_up_matlab_engine_async()
            if os.path.getsize(data_file_path) == 0:
                QMessageBox.warning(self, "Empty TXT File", "The selected TXT file has no valid data rows.")
                self.ok_button.setEnabled(False)
                return

            start_row, end_row = calculate_txt_row_range(
                self.base_offset or 0, self.first_n_latency, self.last_n_latency + 30.0, self.fs
            )

            estimated_total_rows = (
                int(self.record_end_time * self.fs)
                if self.record_end_time is not None and np.isfinite(self.record_end_time)
                else end_row
            )
            self.total_rows = max(end_row, estimated_total_rows)
            start_row = max(1, start_row)
            if start_row >= end_row:
                QMessageBox.warning(self, "Invalid Row Range",
                    f"Calculated row range is invalid (start row {start_row} >= end row {end_row}).\nPlease check whether the sleep-stage file matches the TXT file.")
                self.ok_button.setEnabled(False)
                return

            self.selected_start_row = start_row
            self.selected_end_row = end_row
            self.result_label.setText(
                f"Final data range:\n"
                f"1. TXT size: {os.path.getsize(data_file_path) / (1024 ** 3):.2f} GB (full row scan skipped)\n"
                f"2. Base offset: {self.base_offset or 0}\n"
                f"3. Rows to keep: {start_row} ~ {end_row}\n"
                f"4. Time range: {(start_row-1)/self.fs:.2f} s ~ {end_row/self.fs:.2f} s\n"
                f"(Sampling rate: {self.fs} Hz)"
            )
        elif data_ext in (".edf", ".set", ".fdt"):
            try:
                data_file_path = resolve_eeg_file_path(data_file_path)
                self.data_file_path = data_file_path
                self.txt_file_path = data_file_path
                data_ext = os.path.splitext(data_file_path)[1].lower()
                raw_header = read_supported_eeg(data_file_path, preload=False, verbose=False)
                self.raw_header = raw_header
                file_type = data_ext[1:].upper()
                self.fs = float(raw_header.info['sfreq'])
                self.total_rows = int(raw_header.n_times)
                total_duration = self.total_rows / self.fs
                start_time = max(0.0, float(self.first_n_latency))
                end_time = min(total_duration, float(self.last_n_latency) + 30.0)
                if start_time >= end_time:
                    QMessageBox.warning(
                        self,
                        "Invalid Time Range",
                        f"Calculated time range is invalid ({start_time:.2f} s >= {end_time:.2f} s).\nPlease check whether the sleep-stage file matches the {file_type} file."
                    )
                    self.ok_button.setEnabled(False)
                    return
                self.selected_start_time = start_time
                self.selected_end_time = end_time
                self.selected_start_row = int(start_time * self.fs) + 1
                self.selected_end_row = int(end_time * self.fs)
                self.result_label.setText(
                    f"Final data range:\n"
                    f"1. {file_type} total duration: {total_duration:.2f} s\n"
                    f"2. N-stage segments: {len(self.sleep_intervals)}\n"
                    f"3. Enclosing time range: {start_time:.2f} s ~ {end_time:.2f} s\n"
                    f"4. Sample range: {self.selected_start_row} ~ {self.selected_end_row}\n"
                    f"(Sampling rate: {self.fs:g} Hz)"
                )
            except Exception as e:
                QMessageBox.critical(self, "EEG Read Failed", f"Failed to read EEG file header:\n{str(e)}")
                self.ok_button.setEnabled(False)
                return
        else:
            QMessageBox.warning(self, "Unsupported Format", f"Only TXT, EDF, SET, or FDT files are supported. Current format: {data_ext}")
            self.ok_button.setEnabled(False)
            return

        self.ok_button.setEnabled(True)


    def on_accept(self):
        if self.selected_start_row is None or self.selected_end_row is None or not self.sleep_intervals:
            QMessageBox.warning(self, "Selection Incomplete", "Please select and calculate both the sleep-stage file and EEG file first.")
            return
        self.accept()
