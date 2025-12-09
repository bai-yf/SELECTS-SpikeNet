# main.py
import sys
import os
import numpy as np
import mne
import torch
import torch.nn.functional as F

# PyQt5 imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QScrollArea, 
                            QSlider, QMessageBox, QProgressDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg
from pyqtgraph import PlotWidget, TextItem, LinearRegionItem

# Import from our modular components
from core.models import FusionModel
from core.features import extract_multi_channel_features
from core.preprocessing import call_matlab_preprocess
from core.metrics import calculate_swi, calculate_high_pred1_ratios
from ui.dialogs import (SpikeConfirmDialog, TimeRowRangeDialog, 
                       SleepStageImportDialog)
from visualization.brain_topography import plot_brain_topography, plot_electrode_montage
from utils.file_io import save_spike_data, load_spike_data

class EEGAutoSaveProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Data Processing Tool (Supports TXT+SET+Sleep Staging)")
        self.setGeometry(100, 100, 1400, 950)
        
        pg.setConfigOption('foreground', 'd')
        pg.setConfigOption('background', 'w')
        
        self.raw = None
        self.current_data = None
        self.process_step = 0
        self.channel_names = ['FP1','FP2','F3','F4','C3','C4','P3','P4',
                            'O1','O2','F7','F8','T3','T4','T5','T6']
        self.fs = 500
        
        self.window_duration = 15
        self.total_samples = 0
        self.samples_per_window = int(self.window_duration * self.fs)
        
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_weight_path = r"D:\bai\interface\code\fold_3_best_weights.pth"
        self.window_sec = 0.2
        self.half_window = int(self.window_sec * self.fs / 2)
        self.input_window_size = 2 * self.half_window

        self.window_center_times = []
        self.window_pred_map = {}
        
        self.artefact_threshold = 11.0
        self.artefact_idx = []
        
        self.original_y_unit_scale = 1.0
        self.current_y_scale = 1.0
        
        self.high_prob_spikes = []
        self.low_prob_spikes = []
        self.region_spike_map = {}
        self.file_path = None
        
        self.swi_value = 0.0
        self.spike_seconds = 0
        self.total_seconds = 0
        
        self.init_ui()
        self.init_model()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left function button panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        left_panel.setStyleSheet("padding: 20px;")
        left_panel.setFixedWidth(320)

        left_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        # 0. Import via Sleep Staging 按钮
        self.import_sleep_btn = QPushButton("0. Import via Sleep Staging")
        self.import_sleep_btn.setFixedSize(310, 55)
        self.import_sleep_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                padding: 12px;
                padding-left: 20px;
                background-color: #FF5722;
                color: black;
                text-align: left;
                border: none;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #FF9E7A;
            }
        """)
        self.import_sleep_btn.clicked.connect(self.import_data_by_sleep_stage)
        left_layout.addWidget(self.import_sleep_btn)

        # 1. Import Raw Data 按钮
        self.import_raw_btn = QPushButton("1. Import Raw Data")
        self.import_raw_btn.setFixedSize(310, 55)
        self.import_raw_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                padding: 12px;
                padding-left: 20px;
                background-color: #2196F3;
                color: black;
                text-align: left;
                border: none;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #7AB6F5;
            }
        """)
        self.import_raw_btn.clicked.connect(self.import_raw_data)
        left_layout.addWidget(self.import_raw_btn)

        # 2. Import Preprocessed Data 按钮
        self.import_processed_btn = QPushButton("2. Import Preprocessed Data")
        self.import_processed_btn.setFixedSize(310, 55)
        self.import_processed_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                padding: 12px;
                padding-left: 20px;
                background-color: #9C27B0;
                color: black;
                text-align: left;
                border: none;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #B77BD2;
            }
        """)
        self.import_processed_btn.clicked.connect(self.import_processed_data)
        left_layout.addWidget(self.import_processed_btn)

        # 3. Run Preprocessing 按钮
        self.preprocess_btn = QPushButton("3. Run Preprocessing")
        self.preprocess_btn.setFixedSize(310, 55)
        self.preprocess_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                padding: 12px;
                padding-left: 20px;
                background-color: #4CAF50;
                color: black;
                text-align: left;
                border: none;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #8ED190;
            }
        """)
        self.preprocess_btn.clicked.connect(self.run_preprocessing)
        self.preprocess_btn.setEnabled(False)
        left_layout.addWidget(self.preprocess_btn)

        # 4. Spike Detection 按钮
        self.spike_detect_btn = QPushButton("4. Spike Detection")
        self.spike_detect_btn.setFixedSize(310, 55)
        self.spike_detect_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                padding: 12px;
                padding-left: 20px;
                background-color: #FF9800;
                color: black;
                text-align: left;
                border: none;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #FFC060;
            }
        """)
        self.spike_detect_btn.clicked.connect(self.run_spike_detection)
        self.spike_detect_btn.setEnabled(False)
        left_layout.addWidget(self.spike_detect_btn)

        # 5. Recalculate SWI 按钮
        self.recalc_swi_btn = QPushButton("5. Recalculate SWI")
        self.recalc_swi_btn.setFixedSize(310, 55)
        self.recalc_swi_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                padding: 12px;
                padding-left: 20px;
                background-color: #F44336;
                color: black;
                text-align: left;
                border: none;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #F88A82;
            }
        """)
        self.recalc_swi_btn.clicked.connect(self.recalculate_swi)
        self.recalc_swi_btn.setEnabled(False)
        left_layout.addWidget(self.recalc_swi_btn)

        # 6. View Brain Topography 按钮
        self.plot_brain_btn = QPushButton("6. View Brain Topography")
        self.plot_brain_btn.setFixedSize(310, 55)
        self.plot_brain_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                padding: 12px;
                padding-left: 20px;
                background-color: #2E7D32;
                color: black;
                text-align: left;
                border: none;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #6DBF71;
            }
        """)
        self.plot_brain_btn.clicked.connect(self.show_brain_topography)
        self.plot_brain_btn.setEnabled(False)
        left_layout.addWidget(self.plot_brain_btn)

        # 7. Save Spike Results 按钮
        self.save_spike_btn = QPushButton("7. Save Spike Results")
        self.save_spike_btn.setFixedSize(310, 55)
        self.save_spike_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                padding: 12px;
                padding-left: 20px;
                background-color: #2196F3;
                color: black;
                text-align: left;
                border: none;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #7AB6F5;
            }
        """)
        self.save_spike_btn.clicked.connect(self.save_spike_click)
        self.save_spike_btn.setEnabled(False)
        left_layout.addWidget(self.save_spike_btn)

        # 8. Load Spike Results 按钮
        self.load_spike_btn = QPushButton("8. Load Spike Results")
        self.load_spike_btn.setFixedSize(310, 55)
        self.load_spike_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                padding: 12px;
                padding-left: 20px;
                background-color: #9C27B0;
                color: black;
                text-align: left;
                border: none;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #B77BD2;
            }
        """)
        self.load_spike_btn.clicked.connect(self.load_spike_click)
        left_layout.addWidget(self.load_spike_btn)

        left_layout.addStretch()
        main_layout.addWidget(left_panel, stretch=1)

        # Right display area
        self.right_panel = QScrollArea()
        self.right_content = QWidget()
        self.right_layout = QVBoxLayout(self.right_content)

        self.plot_widget = PlotWidget(title="EEG Waveform Display (Green=High-probability Spike, Red=Low-probability Spike, Click Red Area to Confirm/Reject)")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMinimumHeight(600)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_plot_click)
        self.right_layout.addWidget(self.plot_widget)

        left_axis = self.plot_widget.getAxis('left')
        left_axis.setStyle(tickLength=0)
        left_axis.setTicks([])
        self.plot_widget.setLabel('left', '')

        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setEnabled(False)
        self.time_slider.setToolTip("Drag to control the 15-second time window")
        self.time_slider.valueChanged.connect(self.update_plot_by_slider)
        self.right_layout.addWidget(QLabel("Time Range Control (15-second window):"))
        self.right_layout.addWidget(self.time_slider)

        # Bottom horizontal layout container
        bottom_container = QWidget()
        bottom_layout = QHBoxLayout(bottom_container)
        bottom_layout.setSpacing(15)

        # SWI indicator label
        self.swi_label = QLabel("【Spike Wave Index (SWI) Indicators】\nWaiting for detection or result loading...")
        self.swi_label.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold;
            margin: 5px; 
            padding: 15px; 
            border: 2px solid #4CAF50; 
            border-radius: 8px; 
            background-color: #f1f8e9; 
            white-space: pre-wrap;
        """)
        self.swi_label.setWordWrap(True)
        self.swi_label.setMinimumWidth(400)
        bottom_layout.addWidget(self.swi_label, stretch=2)

        # Status label
        self.status_label = QLabel("Current Status: Waiting for data import | Model loading...")
        self.status_label.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold;
            margin: 5px; 
            padding: 10px; 
            border: 1px solid #eee; 
            border-radius: 5px;
        """)
        self.status_label.setWordWrap(True)
        bottom_layout.addWidget(self.status_label, stretch=1)

        # Info label
        self.info_label = QLabel("Tips:\n1. Supports importing TXT or SET format raw data\n2. Automatically extract N1/N2/N3 stage data via 'Import via Sleep Staging'\n3. Red areas can be clicked after detection: Confirm to turn green, Reject to remove\n4. Click 'Recalculate SWI' after adjustment to update indicators\n5. Click 'View Brain Topography' after spike detection to analyze spatial distribution")
        self.info_label.setStyleSheet('''
            font-size: 18px; 
            font-weight: bold;
            margin: 5px; 
            padding: 15px; 
            border: 2px solid #4CAF50; 
            border-radius: 8px; 
            background-color: #f1f8e9; 
            white-space: pre-wrap;
        ''')
        self.info_label.setWordWrap(True)
        bottom_layout.addWidget(self.info_label, stretch=3)

        self.right_layout.addWidget(bottom_container)

        self.right_panel.setWidget(self.right_content)
        self.right_panel.setWidgetResizable(True)
        main_layout.addWidget(self.right_panel, stretch=4)

    def show_brain_topography(self):
        if self.process_step != 3 or not self.window_center_times or not self.window_pred_map:
            QMessageBox.warning(self, "Conditions Not Met", "Please complete spike detection first to ensure window prediction results exist!")
            return
        
        target_channels = ['T3', 'T4', 'C3', 'C4', 'P3', 'P4', 'T5', 'T6']
        
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
            QMessageBox.critical(self, "Statistics Failed", f"Error calculating ratios: {str(e)}")
            return
        
        channel_counts = {ch: {'true_ratio': ratio} for ch, ratio in channel_ratios.items()}
        
        brain_fig = plot_brain_topography(
            raw_data=self.current_data,
            channel_names=self.channel_names,
            target_channels=target_channels,
            channel_counts=channel_counts,
            sfreq=self.fs
        )
        brain_dialog = QDialog(self)
        brain_dialog.setWindowTitle("Brain Topography")
        brain_dialog.setGeometry(100, 100, 800, 700)
        brain_layout = QVBoxLayout(brain_dialog)
        brain_canvas = FigureCanvas(brain_fig)
        brain_layout.addWidget(brain_canvas)
        brain_btn = QDialogButtonBox(QDialogButtonBox.Close)
        brain_btn.rejected.connect(brain_dialog.close)
        brain_layout.addWidget(brain_btn)
        
        montage_fig = plot_electrode_montage(
            raw_data=self.current_data,
            channel_names=self.channel_names,
            target_channels=target_channels,
            channel_counts=channel_counts,
            sfreq=self.fs
        )
        montage_dialog = QDialog(self)
        montage_dialog.setWindowTitle("Electrode Montage")
        montage_dialog.setGeometry(950, 100, 800, 700)
        montage_layout = QVBoxLayout(montage_dialog)
        montage_canvas = FigureCanvas(montage_fig)
        montage_layout.addWidget(montage_canvas)
        montage_btn = QDialogButtonBox(QDialogButtonBox.Close)
        montage_btn.rejected.connect(montage_dialog.close)
        montage_layout.addWidget(montage_btn)
        
        brain_dialog.show()
        montage_dialog.show()

    def import_data_by_sleep_stage(self):
        sleep_dialog = SleepStageImportDialog(self)
        if sleep_dialog.exec_() != QDialog.Accepted:
            return
        
        txt_file_path = sleep_dialog.txt_file_path
        start_row = sleep_dialog.selected_start_row
        end_row = sleep_dialog.selected_end_row
        total_rows = sleep_dialog.total_rows
        self.fs = sleep_dialog.fs
        
        try:
            all_lines = []
            with open(txt_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, start=1):
                    if start_row <= line_num <= end_row:
                        line = line.strip()
                        if line:
                            all_lines.append(line)
            
            if not all_lines:
                raise ValueError(f"No valid data in selected row range ({start_row}-{end_row} rows)")
            
            data_list = []
            over_16_count = 0
            invalid_count = 0
            over_16_samples = []
            invalid_samples = []
            
            for line_num, line in enumerate(all_lines, start=start_row):
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
            
            data_np = np.array(data_list).T
            if data_np.shape[0] != 16:
                raise ValueError(f"Incorrect total channel count (16 required, {data_np.shape[0]} found)")
            
            info = mne.create_info(
                ch_names=self.channel_names,
                sfreq=self.fs,
                ch_types=['eeg']*16
            )
            self.raw = mne.io.RawArray(data_np, info)
            self.file_path = txt_file_path
            
            self.current_data = self.raw.get_data()
            self.current_data = (self.current_data - np.mean(self.current_data, axis=1, keepdims=True)) / \
                                (np.std(self.current_data, axis=1, keepdims=True) + 1e-6)
            
            self.process_step = 1
            self.total_samples = self.current_data.shape[1]
            max_slider_value = max(0, self.total_samples - self.samples_per_window)
            self.time_slider.setMaximum(max_slider_value)
            self.time_slider.setEnabled(True)
            
            self.high_prob_spikes.clear()
            self.low_prob_spikes.clear()
            self.region_spike_map.clear()
            self.window_center_times.clear()
            self.window_pred_map.clear()
            self.swi_value = 0.0
            self.spike_seconds = 0
            self.total_seconds = 0
            self.swi_label.setText("【Spike Wave Index (SWI) Indicators】\nWaiting for detection or result loading...")
            self.save_spike_btn.setEnabled(False)
            self.recalc_swi_btn.setEnabled(False)
            self.plot_brain_btn.setEnabled(False)
            
            self.update_plot_by_slider(0)
            self.status_label.setText(f"Current Status: TXT data imported via sleep staging | Model loaded successfully")
            self.info_label.setText(
                f"✅ Imported successfully via sleep staging!\n"
                f"File: {os.path.basename(txt_file_path)}\n"
                f"Selected range: {start_row} rows ~ {end_row} rows (total {len(all_lines)} rows)\n"
                f"Channels: 16 | Sampling rate: {self.fs}Hz\n"
                f"Total data duration: {self.total_samples/self.fs:.2f} seconds\n"
                f"Next step: Click 'Run Preprocessing'"
            )
            self.preprocess_btn.setEnabled(True)
            self.spike_detect_btn.setEnabled(False)
            
            messages = []
            if over_16_count > 0:
                if over_16_count <= 5:
                    rows_str = ", ".join(map(str, over_16_samples))
                else:
                    rows_str = ", ".join(map(str, over_16_samples[:5])) + f" and {over_16_count} other rows"
                messages.append(f"Detected {over_16_count} rows with more than 16 channels: {rows_str}")
                messages.append("Automatically truncated to first 16 channels for analysis")
            
            if invalid_count > 0:
                if invalid_count <= 5:
                    rows_str = ", ".join(map(str, invalid_samples))
                else:
                    rows_str = ", ".join(map(str, invalid_samples[:5])) + f" and {invalid_count} other rows"
                messages.append(f"Found {invalid_count} rows with less than 16 channels: {rows_str}")
                messages.append("These rows have been skipped and not included in analysis")
            
            if messages:
                icon = QMessageBox.Warning if invalid_count > 0 else QMessageBox.Information
                QMessageBox(icon, "Data Processing Tip", "\n".join(messages), QMessageBox.Ok, self).exec_()
        
        except Exception as e:
            QMessageBox.critical(self, "Sleep Staging Import Failed", f"Error: {str(e)}")
            self.status_label.setText("Current Status: Sleep staging import failed | Model loaded successfully")
            self.info_label.setText(f"❌ Error: {str(e)}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.zoom_y_axis(scale_factor=1.2)
        elif event.key() == Qt.Key_Minus:
            self.zoom_y_axis(scale_factor=0.8)
        super().keyPressEvent(event)

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
        self.plot_widget.setLabel('left', f'Amplitude (μV × {scaled_unit:.2f})')

    def on_plot_click(self, event):
        if self.process_step != 3:
            return
        
        for region, spike_time in self.region_spike_map.items():
            region_x = region.getRegion()
            click_x = self.plot_widget.plotItem.vb.mapSceneToView(event.scenePos()).x()
            
            if region_x[0] <= click_x <= region_x[1] and region.brush.color().red() == 255:
                dialog = SpikeConfirmDialog(spike_time, self)
                result = dialog.exec_()
                
                if result == QDialog.Accepted:
                    self.confirm_low_spike(spike_time)
                else:
                    self.reject_low_spike(spike_time)
                break

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
        
        QMessageBox.information(self, "Operation Successful", f"Confirmed spike at {spike_time:.3f} seconds, turned to green region")

    def reject_low_spike(self, spike_time):
        for i, (t, prob) in enumerate(self.low_prob_spikes):
            if abs(t - spike_time) < 1e-3:
                del self.low_prob_spikes[i]
                break
        else:
            return
        
        self.update_plot_by_slider(self.time_slider.value())
        
        QMessageBox.information(self, "Operation Successful", f"Rejected spike at {spike_time:.3f} seconds, removed red region")

    def recalculate_swi(self):
        if self.total_samples == 0:
            QMessageBox.warning(self, "Calculation Failed", "No data available to calculate SWI!")
            return
        
        all_spike_times = [t for t, _ in self.high_prob_spikes] + [t for t, _ in self.low_prob_spikes]
        total_duration = self.total_samples / self.fs
        
        self.swi_value, self.spike_seconds, self.total_seconds = calculate_swi(
            all_spike_times, total_duration
        )
        
        self.swi_label.setText(
            f"【Spike Wave Index (SWI) Indicators】\n"
            f"1. SWI value: {self.swi_value}%\n"
            f"2. Total high-probability spikes: {len(self.high_prob_spikes)}\n"
            f"3. Total low-probability spikes: {len(self.low_prob_spikes)}\n"
            f"4. Seconds with spikes: {self.spike_seconds} seconds\n"
            f"5. Total data seconds: {self.total_seconds} seconds\n"
            f"6. Original data duration: {total_duration:.2f} seconds"
        )
        
        self.status_label.setText(
            f"Current Status: SWI recalculated | {len(self.high_prob_spikes)} high-probability spikes, {len(self.low_prob_spikes)} low-probability spikes"
        )

    def init_model(self):
        try:
            self.model = FusionModel(
                num_channels=16,
                time_steps=self.input_window_size,
                num_classes=2
            ).to(self.device)
            
            checkpoint = torch.load(self.model_weight_path, map_location=self.device)
            if "module." in list(checkpoint.keys())[0]:
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            self.status_label.setText("Current Status: Waiting for data import | Model loaded successfully")
            
        except FileNotFoundError:
            QMessageBox.critical(self, "Model Loading Failed", f"Weights file not found!\nPath: {self.model_weight_path}")
            self.status_label.setText("Current Status: Waiting for data import | Model loading failed (missing weights)")
            self.spike_detect_btn.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Model Loading Failed", f"Error: {str(e)}\nPossible mismatch between weights and model structure")
            self.status_label.setText("Current Status: Waiting for data import | Model loading failed (structure mismatch)")
            self.spike_detect_btn.setEnabled(False)

    def import_raw_data(self):
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Raw EEG Data (TXT/SET)", os.getcwd(), "EEG Files (*.txt *.set);;All Files (*.*)"
        )
        
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Tip", "No valid file selected or file does not exist!")
            return
        
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == ".txt":
                all_lines = []
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_lines.append(line)
                total_rows = len(all_lines)
                if total_rows == 0:
                    raise ValueError("TXT file is empty, no valid data")

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
                        rows_str = ", ".join(map(str, over_16_samples[:5])) + f" and {over_16_count} other rows"
                    messages.append(f"Detected {over_16_count} rows with more than 16 channels: {rows_str}")
                    messages.append("Automatically truncated to first 16 channels for analysis")
                
                if invalid_count > 0:
                    if invalid_count <= 5:
                        rows_str = ", ".join(map(str, invalid_samples))
                    else:
                        rows_str = ", ".join(map(str, invalid_samples[:5])) + f" and {invalid_count} other rows"
                    messages.append(f"Found {invalid_count} rows with less than 16 channels: {rows_str}")
                    messages.append("These rows have been skipped and not included in analysis")
                
                if messages:
                    icon = QMessageBox.Warning if invalid_count > 0 else QMessageBox.Information
                    QMessageBox(
                        icon,
                        "Data Processing Tip",
                        "\n".join(messages),
                        QMessageBox.Ok,
                        self
                    ).exec_()

                data_np = np.array(data_list).T
                if data_np.shape[0] > 16:
                    data_np = data_np[:16, :]
                    QMessageBox.information(
                        self,
                        "Data Processing Tip",
                        "Total channels exceed 16, automatically truncated to first 16 channels",
                        QMessageBox.Ok
                    )
                elif data_np.shape[0] != 16:
                    raise ValueError(f"Incorrect total channel count (16 required, {data_np.shape[0]} found)")

                info = mne.create_info(
                    ch_names=self.channel_names,
                    sfreq=self.fs,
                    ch_types=['eeg']*16
                )
                self.raw = mne.io.RawArray(data_np, info)

            elif file_ext == ".set":
                self.raw = mne.io.read_raw_eeglab(file_path, preload=True)
            else:
                raise ValueError(f"Unsupported format: {file_ext} (only TXT/SET supported)")
            
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
            
            self.high_prob_spikes.clear()
            self.low_prob_spikes.clear()
            self.region_spike_map.clear()
            self.window_center_times.clear()
            self.window_pred_map.clear()
            self.swi_value = 0.0
            self.spike_seconds = 0
            self.total_seconds = 0
            self.swi_label.setText("【Spike Wave Index (SWI) Indicators】\nWaiting for detection or result loading...")
            self.save_spike_btn.setEnabled(False)
            self.recalc_swi_btn.setEnabled(False)
            self.plot_brain_btn.setEnabled(False)
            
            file_type = "TXT" if file_ext == ".txt" else "SET"
            self.status_label.setText(f"Current Status: {file_type} raw data imported | Model loaded successfully")
            self.info_label.setText(
                f"✅ {file_type} raw data imported successfully!\n"
                f"File: {os.path.basename(file_path)}\n"
                f"Channels: 16 | Sampling rate: {self.fs}Hz\n"
                f"Total data duration: {self.total_samples/self.fs:.2f} seconds\n"
                f"Next step: Click 'Run Preprocessing'"
            )
            self.preprocess_btn.setEnabled(True)
            self.spike_detect_btn.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Error reading raw file:\n{str(e)}")
            self.status_label.setText(f"Current Status: {file_type} raw data import failed | Model loaded successfully")
            self.info_label.setText(f"❌ Error: {str(e)}")
            
    def import_processed_data(self):
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Preprocessed EEG Data (.set)", os.getcwd(), "EEG Files (*.set);;All Files (*.*)"
        )
        
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Tip", "No valid file selected or file does not exist!")
            return
        
        try:
            self.raw = mne.io.read_raw_eeglab(file_path, preload=True)
            self.current_data = self.raw.get_data()
            self.current_data = (self.current_data - np.mean(self.current_data, axis=1, keepdims=True)) / \
                                (np.std(self.current_data, axis=1, keepdims=True) + 1e-6)
            
            self.process_step = 2
            self.total_samples = self.current_data.shape[1]
            max_slider_value = max(0, self.total_samples - self.samples_per_window)
            self.time_slider.setMaximum(max_slider_value)
            self.time_slider.setEnabled(True)
            
            self.update_plot_by_slider(0)
            
            self.high_prob_spikes.clear()
            self.low_prob_spikes.clear()
            self.region_spike_map.clear()
            self.window_center_times.clear()
            self.window_pred_map.clear()
            self.swi_value = 0.0
            self.spike_seconds = 0
            self.total_seconds = 0
            self.swi_label.setText("【Spike Wave Index (SWI) Indicators】\nWaiting for detection or result loading...")
            self.save_spike_btn.setEnabled(False)
            self.recalc_swi_btn.setEnabled(False)
            self.plot_brain_btn.setEnabled(False)
            
            self.status_label.setText("Current Status: Preprocessed data imported | Model loaded successfully")
            self.info_label.setText(
                f"✅ Preprocessed data imported successfully!\n"
                f"File: {os.path.basename(file_path)}\n"
                f"Channels: 16 | Sampling rate: {self.fs}Hz\n"
                f"Total data duration: {self.total_samples/self.fs:.2f} seconds\n"
                f"Next step: Click 'Spike Detection'"
            )
            self.preprocess_btn.setEnabled(False)
            self.spike_detect_btn.setEnabled(True if self.model is not None else False)
            
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Error reading preprocessed file:\n{str(e)}")
            self.status_label.setText("Current Status: Preprocessed data import failed | Model loaded successfully")
            self.info_label.setText(f"❌ Error: {str(e)}")

    def run_preprocessing(self):
        if self.process_step != 1 or self.raw is None:
            QMessageBox.warning(self, "Preprocessing Conditions Not Met", "Please import raw data first!")
            return

        try:
            progress = QProgressDialog("Running MATLAB preprocessing...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Preprocessing Progress")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            progress.setValue(10)

            if self.file_path.lower().endswith('.txt'):
                processed_set_path = call_matlab_preprocess(self.file_path, fs=self.fs)
                progress.setValue(60)
                self.raw = mne.io.read_raw_eeglab(processed_set_path, preload=True)
            else:
                processed_set_path = self.file_path

            progress.setValue(80)
            self.current_data = self.raw.get_data()
            self.current_data = (self.current_data - np.mean(self.current_data, axis=1, keepdims=True)) / \
                                (np.std(self.current_data, axis=1, keepdims=True) + 1e-6)

            self.process_step = 2
            progress.setValue(100)
            progress.close()

            self.status_label.setText("Current Status: Preprocessing completed | Model loaded successfully")
            self.info_label.setText(
                f"✅ Preprocessing completed successfully!\n"
                f"Processed file saved to: {processed_set_path}\n"
                f"Channels: 16 | Sampling rate: {self.fs}Hz\n"
                f"Total data duration: {self.total_samples/self.fs:.2f} seconds\n"
                f"Next step: Click 'Spike Detection'"
            )
            self.preprocess_btn.setEnabled(False)
            self.spike_detect_btn.setEnabled(True if self.model is not None else False)

        except Exception as e:
            QMessageBox.critical(self, "Preprocessing Failed", f"Error during preprocessing:\n{str(e)}")
            self.status_label.setText("Current Status: Preprocessing failed | Model loaded successfully")
            self.info_label.setText(f"❌ Preprocessing error: {str(e)}")

    def run_spike_detection(self):
        if self.process_step != 2 or self.model is None or self.current_data is None:
            QMessageBox.warning(self, "Detection Conditions Not Met", "Please complete preprocessing and ensure model is loaded!")
            return

        try:
            progress = QProgressDialog("Running spike detection...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Spike Detection Progress")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            progress.setValue(10)

            self.high_prob_spikes.clear()
            self.low_prob_spikes.clear()
            self.region_spike_map.clear()
            self.window_center_times.clear()
            self.window_pred_map.clear()

            total_samples = self.current_data.shape[1]
            valid_windows = []

            progress.setValue(20)
            for i in range(self.half_window, total_samples - self.half_window, self.input_window_size):
                if progress.wasCanceled():
                    raise InterruptedError("Spike detection canceled by user")

                window_data = self.current_data[:, i - self.half_window:i + self.half_window]
                if np.isnan(window_data).any() or np.std(window_data) < 1e-6:
                    continue

                features = extract_multi_channel_features(window_data)
                features = np.expand_dims(features, axis=0)
                features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    output = self.model(features_tensor)
                    pred_prob = F.softmax(output, dim=1)[:, 1].cpu().numpy()[0]
                    window_center_time = i / self.fs

                    valid_windows.append((window_center_time, pred_prob))

                progress.setValue(min(20 + int((i / total_samples) * 70), 90))

            progress.setValue(90)
            for window_center_time, pred_prob in valid_windows:
                self.window_center_times.append(window_center_time)
                self.window_pred_map[window_center_time] = 1 if pred_prob >= 0.5 else 0

                if pred_prob >= 0.8:
                    self.high_prob_spikes.append((window_center_time, pred_prob))
                elif 0.5 <= pred_prob < 0.8:
                    self.low_prob_spikes.append((window_center_time, pred_prob))

            self.high_prob_spikes.sort(key=lambda x: x[0])
            self.low_prob_spikes.sort(key=lambda x: x[0])
            self.window_center_times.sort()

            total_duration = self.total_samples / self.fs
            self.swi_value, self.spike_seconds, self.total_seconds = calculate_swi(
                [t for t, _ in self.high_prob_spikes] + [t for t, _ in self.low_prob_spikes],
                total_duration
            )

            self.swi_label.setText(
                f"【Spike Wave Index (SWI) Indicators】\n"
                f"1. SWI value: {self.swi_value}%\n"
                f"2. Total high-probability spikes (≥0.8): {len(self.high_prob_spikes)}\n"
                f"3. Total low-probability spikes (0.5-0.8): {len(self.low_prob_spikes)}\n"
                f"4. Seconds with spikes: {self.spike_seconds} seconds\n"
                f"5. Total data seconds: {self.total_seconds} seconds\n"
                f"6. Original data duration: {total_duration:.2f} seconds"
            )

            self.update_plot_by_slider(self.time_slider.value())
            progress.setValue(100)
            progress.close()

            self.process_step = 3
            self.status_label.setText(
                f"Current Status: Spike detection completed | {len(self.high_prob_spikes)} high-probability, {len(self.low_prob_spikes)} low-probability spikes"
            )
            self.info_label.setText(
                f"✅ Spike detection completed!\n"
                f"Total detected spikes: {len(self.high_prob_spikes) + len(self.low_prob_spikes)}\n"
                f"- High-probability (≥0.8): {len(self.high_prob_spikes)} (green regions)\n"
                f"- Low-probability (0.5-0.8): {len(self.low_prob_spikes)} (red regions)\n"
                f"Operations available:\n"
                f"1. Click red regions to confirm/reject low-probability spikes\n"
                f"2. Click 'Recalculate SWI' after adjustment\n"
                f"3. Click 'Save Spike Results' to export data\n"
                f"4. Click 'View Brain Topography' to analyze spatial distribution"
            )
            self.recalc_swi_btn.setEnabled(True)
            self.save_spike_btn.setEnabled(True)
            self.plot_brain_btn.setEnabled(True)

        except InterruptedError:
            self.status_label.setText("Current Status: Spike detection canceled by user | Model loaded successfully")
            self.info_label.setText("⚠️ Spike detection was canceled, no results generated")
        except Exception as e:
            QMessageBox.critical(self, "Spike Detection Failed", f"Error during spike detection:\n{str(e)}")
            self.status_label.setText("Current Status: Spike detection failed | Model loaded successfully")
            self.info_label.setText(f"❌ Detection error: {str(e)}")

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
        self.region_spike_map.clear()
        self.plot_widget.setTitle(f"EEG waveform | time range：{window_start}-{window_end}s (15s window)")
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        max_amplitude = np.max(np.abs(plot_data)) if plot_data.size > 0 else 1
        offset = max_amplitude * 1.5
        highlight_channels = {4, 5, 12, 13}
        
        for i in range(plot_data.shape[0]):
            channel_data = plot_data[i]
            color = 'red' if i in highlight_channels else 'blue'
            offset_data = channel_data + i * offset
            
            self.plot_widget.plot(
                times, offset_data,
                label=self.channel_names[i],
                pen=pg.mkPen(color=color, width=2)
            )
            
            if len(times) > 0 and len(offset_data) > 0:
                text = TextItem(self.channel_names[i], anchor=(1, 0.5), color=(0, 0, 0))
                text.setFont(QFont("Arial", 10))
                text.setPos(times[0], offset_data[0])
                self.plot_widget.addItem(text)
        
        current_high_spikes = [s for s in self.high_prob_spikes if window_start <= s[0] <= window_end]
        for spike_time, _ in current_high_spikes:
            region = LinearRegionItem(
                values=[spike_time - 0.1, spike_time + 0.1],
                brush=pg.mkBrush(color=(0, 255, 0, 40)),
                movable=False
            )
            self.plot_widget.addItem(region)
            self.region_spike_map[region] = spike_time
        
        current_low_spikes = [s for s in self.low_prob_spikes if window_start <= s[0] <= window_end]
        for spike_time, _ in current_low_spikes:
            region = LinearRegionItem(
                values=[spike_time - 0.1, spike_time + 0.1],
                brush=pg.mkBrush(color=(255, 0, 0, 40)),
                movable=False
            )
            self.plot_widget.addItem(region)
            self.region_spike_map[region] = spike_time

        if len(times) > 0:
            expected_end = times[0] + self.window_duration
            self.plot_widget.setXRange(times[0], expected_end, padding=0)
        
        if plot_data.size > 0:
            self.plot_widget.setYRange(-offset * 0.5, plot_data.shape[0] * offset + offset * 0.5)
            self.current_y_scale = 1.0
            self.plot_widget.setLabel('left', 'Amplitude (μV)')

        self.plot_widget.setLabel('bottom', 'Time ', 's', font=QFont("Arial", 12))

        legend = self.plot_widget.addLegend(offset=(10, 10), labelTextSize='10pt')
        legend.setParentItem(self.plot_widget.graphicsItem())

    def save_spike_click(self):
        if self.process_step != 3:
            QMessageBox.warning(self, "Save Conditions Not Met", "Please complete spike detection first!")
            return

        all_spike_times = [t for t, _ in self.high_prob_spikes] + [t for t, _ in self.low_prob_spikes]
        save_success = save_spike_data(all_spike_times, self.current_data, self.fs)
        if save_success:
            self.status_label.setText(
                f"Current Status: Spike results saved | {len(self.high_prob_spikes)} high-probability, {len(self.low_prob_spikes)} low-probability spikes"
            )

    def load_spike_click(self):
        loaded_data = load_spike_data()
        if loaded_data is None:
            return

        try:
            self.current_data = loaded_data["eeg_data"]
            self.fs = loaded_data["fs"]
            self.total_samples = self.current_data.shape[1]
            self.process_step = 3

            self.high_prob_spikes = []
            self.low_prob_spikes = []
            spike_times = loaded_data["spike_times"]

            total_duration = self.total_samples / self.fs
            self.swi_value, self.spike_seconds, self.total_seconds = calculate_swi(spike_times, total_duration)

            for t in spike_times:
                self.high_prob_spikes.append((t, 1.0))

            self.swi_label.setText(
                f"【Spike Wave Index (SWI) Indicators】\n"
                f"1. SWI value: {self.swi_value}%\n"
                f"2. Total loaded spikes: {len(spike_times)}\n"
                f"3. Seconds with spikes: {self.spike_seconds} seconds\n"
                f"4. Total data seconds: {self.total_seconds} seconds\n"
                f"5. Original data duration: {total_duration:.2f} seconds\n"
                f"6. Data saved at: {loaded_data['save_time']}"
            )

            max_slider_value = max(0, self.total_samples - self.samples_per_window)
            self.time_slider.setMaximum(max_slider_value)
            self.time_slider.setEnabled(True)
            self.update_plot_by_slider(0)

            self.status_label.setText(
                f"Current Status: Spike results loaded | {len(spike_times)} spikes loaded"
            )
            self.info_label.setText(
                f"✅ Spike results loaded successfully!\n"
                f"Loaded from file: {loaded_data.get('save_path', 'unknown')}\n"
                f"Total spikes: {len(spike_times)}\n"
                f"Sampling rate: {self.fs}Hz | Total duration: {total_duration:.2f} seconds\n"
                f"Next step: Click 'Recalculate SWI' if adjustments are made"
            )
            self.recalc_swi_btn.setEnabled(True)
            self.save_spike_btn.setEnabled(True)
            self.plot_brain_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Error loading spike data:\n{str(e)}")
            self.status_label.setText("Current Status: Spike results load failed | Model loaded successfully")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGAutoSaveProcessor()
    window.show()
    sys.exit(app.exec_())