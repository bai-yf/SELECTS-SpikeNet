
"""Lightweight spike-result persistence and backward-compatible loading."""
import os
import time

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from utils.eeg_io import load_eeg_from_source, resolve_eeg_file_path

def save_spike_data(spike_times, eeg_data, fs, default_filename="spike_results.npz",
                    high_prob_spikes=None, low_prob_spikes=None, save_path=None,
                    show_dialog=True, parent=None, total_record_duration=None,
                    analyzable_duration=None, excluded_duration=None,
                    spike_detection_elapsed=None, source_file_path=None,
                    source_intervals=None, source_invert=False,
                    source_channel_names=None):
    if save_path is None:
        save_path, _ = QFileDialog.getSaveFileName(
            parent, "Save Spike Detection Results", default_filename, "NPZ files (*.npz);;All files (*.*)"
        )
    if not save_path:
        return None
    try:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        high_prob_spikes = high_prob_spikes if high_prob_spikes is not None else []
        low_prob_spikes = low_prob_spikes if low_prob_spikes is not None else []
        if not source_file_path:
            raise ValueError(
                "Cannot create a lightweight result because the EEG source path is unavailable. "
                "Please import/preprocess the EEG again before saving."
            )
        source_file_path = os.path.abspath(source_file_path)
        if not os.path.exists(source_file_path):
            raise FileNotFoundError(f"EEG source file does not exist: {source_file_path}")
        if os.path.splitext(resolve_eeg_file_path(source_file_path))[1].lower() not in (".edf", ".set"):
            raise ValueError(
                "Lightweight results require an EDF/SET EEG source. "
                "Please run preprocessing (or import a preprocessed SET) before saving."
            )

        np.savez_compressed(
            save_path,
            spike_times=np.array(spike_times, dtype=float),
            high_prob_spikes=np.array(high_prob_spikes, dtype=float),
            low_prob_spikes=np.array(low_prob_spikes, dtype=float),
            fs=fs,
            result_format_version=2,
            source_file_path=source_file_path,
            source_intervals=np.asarray(source_intervals or [], dtype=float).reshape(-1, 2),
            source_invert=bool(source_invert),
            source_channel_names=np.asarray(source_channel_names or [], dtype=str),
            total_record_duration=total_record_duration,
            analyzable_duration=analyzable_duration,
            excluded_duration=excluded_duration,
            spike_detection_elapsed=spike_detection_elapsed,
            save_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            **{
                "总记录时长": total_record_duration,
                "可分析时长": analyzable_duration,
                "被排除时长": excluded_duration,
                "棘波检测用时": spike_detection_elapsed,
            }
        )
        if show_dialog:
            QMessageBox.information(parent, "Saved", f"Spike data saved to:\n{save_path}")
        return save_path
    except Exception as e:
        QMessageBox.critical(parent, "Save Failed", f"Error: {str(e)}")
        return None


def load_spike_data():
    load_path, _ = QFileDialog.getOpenFileName(
        None, "Load Spike Detection Results", "", "NPZ files (*.npz);;All files (*.*)"
    )
    if not load_path:
        return None
    try:
        with np.load(load_path, allow_pickle=True) as data:
            if "high_prob_spikes" in data.files or "low_prob_spikes" in data.files:
                high_prob_spikes = data["high_prob_spikes"].tolist() if "high_prob_spikes" in data.files else []
                low_prob_spikes = data["low_prob_spikes"].tolist() if "low_prob_spikes" in data.files else []
            else:
                # 兼容旧版本结果：旧文件只保存合并后的spike_times，无法区分高/低概率，只能按高概率加载。
                high_prob_spikes = [(float(t), 1.0) for t in data["spike_times"].tolist()]
                low_prob_spikes = []

            spike_times = data["spike_times"].tolist()
            fs = data["fs"].item()
            save_time = data["save_time"].item()

            if "eeg_data" in data.files:
                # Backward compatibility with the original, large result format.
                eeg_data = data["eeg_data"]
                source_file_path = ""
                source_intervals = []
                source_invert = False
                source_channel_names = []
            else:
                source_file_path = str(data["source_file_path"].item())
                source_intervals = data["source_intervals"].tolist() if "source_intervals" in data.files else []
                source_invert = bool(data["source_invert"].item()) if "source_invert" in data.files else False
                source_channel_names = data["source_channel_names"].tolist() if "source_channel_names" in data.files else []

                if not os.path.exists(source_file_path):
                    QMessageBox.warning(
                        None,
                        "EEG Source Not Found",
                        f"The EEG source recorded in this result cannot be found:\n{source_file_path}\n\n"
                        "Please locate the source EEG file."
                    )
                    replacement_path, _ = QFileDialog.getOpenFileName(
                        None,
                        "Locate EEG Source",
                        os.path.dirname(load_path),
                        "EEG files (*.edf *.set *.fdt);;All files (*.*)"
                    )
                    if not replacement_path:
                        return None
                    source_file_path = resolve_eeg_file_path(replacement_path)

                eeg_data, fs = load_eeg_from_source(
                    source_file_path,
                    source_intervals,
                    source_invert,
                    source_channel_names,
                    fs
                )

        return {
            "spike_times": spike_times,
            "high_prob_spikes": high_prob_spikes,
            "low_prob_spikes": low_prob_spikes,
            "eeg_data": eeg_data,
            "fs": fs,
            "save_time": save_time,
            "load_path": load_path,
            "source_file_path": source_file_path,
            "source_intervals": source_intervals,
            "source_invert": source_invert,
            "source_channel_names": source_channel_names,
        }
    except Exception as e:
        QMessageBox.critical(None, "Load Failed", f"Error: {str(e)}\nPlease confirm this is a valid spike result file.")
        return None
