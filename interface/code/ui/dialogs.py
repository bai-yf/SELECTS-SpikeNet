# ui/dialogs.py
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QFileDialog, QSpinBox, QTimeEdit,
                            QDialogButtonBox, QWidget, QMessageBox)
from PyQt5.QtCore import QTime
import pandas as pd
import re
import os

class SpikeConfirmDialog(QDialog):
    def __init__(self, spike_time, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Spike")
        self.setFixedSize(300, 150)
        
        layout = QVBoxLayout()
        self.label = QLabel(f"Confirm if the region at {spike_time:.3f} seconds is a spike?")
        layout.addWidget(self.label)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Ok).setText("Confirm (Turn Green)")
        button_box.button(QDialogButtonBox.Cancel).setText("Reject (Remove Region)")
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

class TimeRowRangeDialog(QDialog):
    def __init__(self, total_rows, fs=500, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Data Range (Row or Time)")
        self.setFixedSize(400, 300)
        self.total_rows = total_rows
        self.fs = fs
        self.total_seconds = total_rows / fs

        self.selected_start_row = 1
        self.selected_end_row = total_rows
        self.selected_start_time = QTime(0, 0, 0)
        self.selected_end_time = self.seconds_to_qtime(self.total_seconds)

        layout = QVBoxLayout()

        self.info_label = QLabel(
            f"Total rows: {total_rows} (Sampling rate {fs}Hz, Total duration {self.format_seconds(self.total_seconds)})\n"
            "Please select the data range to keep:"
        )
        layout.addWidget(self.info_label)

        row_group = QWidget()
        row_layout = QVBoxLayout(row_group)
        row_layout.addWidget(QLabel("--- Row Range Input ---"))
        
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
        time_layout.addWidget(QLabel("--- Time Range Input (HH:MM:SS) ---"))
        
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
        button_box.button(QDialogButtonBox.Ok).setText("Confirm Selection")
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
            QMessageBox.warning(self, "Input Error", "Start row cannot be greater than end row!")
            return
        self.accept()

def parse_sleep_stage_file(sleep_file_path):
    try:
        sleep_df = pd.read_csv(sleep_file_path)
        n_stages = sleep_df[sleep_df['type'].str.upper().isin(['N1', 'N2', 'N3'])]
        
        if len(n_stages) == 0:
            QMessageBox.warning(None, "No Sleep Stage Data", "No N1/N2/N3 stages found in sleep staging file!")
            return None, None
        
        first_n_latency = n_stages['latency'].min()
        last_n_latency = n_stages['latency'].max()
        return first_n_latency, last_n_latency
    
    except Exception as e:
        QMessageBox.critical(None, "Sleep Stage File Parsing Failed", f"Error: {str(e)}\nPlease confirm file format is CSV (contains latency,type columns)")
        return None, None

def extract_base_offset_from_filename(sleep_filename):
    try:
        pattern = r'_(\d+)_\d+_'
        match = re.search(pattern, sleep_filename)
        
        if not match:
            num_pattern = r'(\d{5,})'
            num_matches = re.findall(num_pattern, sleep_filename)
            if not num_matches:
                raise ValueError("No base offset number found in filename (requires at least 5 digits)")
            base_offset = int(num_matches[0])
        else:
            base_offset = int(match.group(1))
        
        return base_offset
    
    except Exception as e:
        QMessageBox.warning(
            None, 
            "Filename Parsing Failed", 
            f"Cannot extract base offset from sleep stage filename: {str(e)}\n"
            f"Please check filename format (e.g., contains '_number_number_' structure, like 'xxx_11567_18947_xxx.csv')"
        )
        return None

def calculate_txt_row_range(base_offset, first_n_latency, last_n_latency, fs=500):
    start_row = int((base_offset + first_n_latency) * fs) + 1
    end_row = int((base_offset + last_n_latency) * fs)
    return start_row, end_row

class SleepStageImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Data Based on Sleep Stages")
        self.setFixedSize(500, 350)
        
        self.txt_file_path = None
        self.selected_start_row = None
        self.selected_end_row = None
        self.total_rows = None
        self.fs = 500
        
        layout = QVBoxLayout()
        
        self.sleep_file_label = QLabel("No sleep staging file selected")
        self.sleep_file_btn = QPushButton("1. Select Sleep Staging CSV File")
        self.sleep_file_btn.clicked.connect(self.select_sleep_file)
        layout.addWidget(self.sleep_file_btn)
        layout.addWidget(self.sleep_file_label)
        
        self.txt_file_label = QLabel("No TXT file selected")
        self.txt_file_btn = QPushButton("2. Select Corresponding EEG TXT File")
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
        button_box.button(QDialogButtonBox.Ok).setText("Confirm Import of This Range")
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
        self.base_offset = None

    def select_sleep_file(self):
        sleep_file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Sleep Staging File", os.getcwd(), "CSV Files (*.csv);;All Files (*.*)"
        )
        if not sleep_file_path:
            return
        
        self.sleep_file_path = sleep_file_path
        sleep_filename = os.path.basename(sleep_file_path)
        self.sleep_file_label.setText(f"Sleep Staging File:\n{sleep_filename}")
        
        self.first_n_latency, self.last_n_latency = parse_sleep_stage_file(sleep_file_path)
        if self.first_n_latency is None or self.last_n_latency is None:
            self.txt_file_btn.setEnabled(False)
            return
        
        self.base_offset = extract_base_offset_from_filename(sleep_filename)
        if self.base_offset is None:
            self.txt_file_btn.setEnabled(False)
            self.result_label.setText("Failed to extract base offset from sleep staging filename, please check format")
            return
        
        self.txt_file_btn.setEnabled(True)
        self.result_label.setText(
            f"Sleep Stage Parsing Results:\n"
            f"1. First N stage latency: {self.first_n_latency} seconds\n"
            f"2. Last N stage latency: {self.last_n_latency} seconds\n"
            f"3. Base offset (extracted from filename): {self.base_offset}\n"
            f"Please select corresponding TXT file to calculate row range"
        )

    def select_txt_file(self):
        txt_file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Corresponding EEG TXT File", os.getcwd(), "TXT Files (*.txt);;All Files (*.*)"
        )
        if not txt_file_path:
            return
        
        self.txt_file_path = txt_file_path
        txt_filename = os.path.basename(txt_file_path)
        self.txt_file_label.setText(f"TXT File:\n{txt_filename}")
        
        with open(txt_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            self.total_rows = sum(1 for line in f if line.strip())
        if self.total_rows == 0:
            QMessageBox.warning(self, "Empty TXT File", "Selected TXT file has no valid data rows!")
            self.ok_button.setEnabled(False)
            return
        
        start_row, end_row = calculate_txt_row_range(
            self.base_offset, self.first_n_latency, self.last_n_latency, self.fs
        )
        
        start_row = max(1, start_row)
        end_row = min(self.total_rows, end_row)
        if start_row >= end_row:
            QMessageBox.warning(self, "Invalid Row Range", 
                f"Calculated row range is invalid (start row {start_row} ≥ end row {end_row})\nPlease check sleep stage and TXT file matching"
            )
            self.ok_button.setEnabled(False)
            return
        
        self.selected_start_row = start_row
        self.selected_end_row = end_row
        self.result_label.setText(
            f"Final Data Range Calculation Results:\n"
            f"1. Total rows in TXT file: {self.total_rows} rows\n"
            f"2. Base offset (from sleep filename): {self.base_offset}\n"
            f"3. Retained row range: {start_row} rows ~ {end_row} rows\n"
            f"4. Corresponding time range: {(start_row-1)/self.fs:.2f} seconds ~ {end_row/self.fs:.2f} seconds\n"
            f"(Sampling rate: {self.fs}Hz)"
        )
        self.ok_button.setEnabled(True)

    def on_accept(self):
        if self.selected_start_row is None or self.selected_end_row is None:
            QMessageBox.warning(self, "Incomplete Selection", "Please complete sleep staging file and TXT file selection and calculation first!")
            return
        self.accept()