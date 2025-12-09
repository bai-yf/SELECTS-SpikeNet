# utils/file_io.py
import numpy as np
import time
from PyQt5.QtWidgets import QFileDialog, QMessageBox

def save_spike_data(spike_times, eeg_data, fs, default_filename="spike_results.npz"):
    save_path, _ = QFileDialog.getSaveFileName(
        None, "Save Spike Detection Results", default_filename, "NPZ Files (*.npz);;All Files (*.*)"
    )
    if not save_path:
        return False
    try:
        np.savez_compressed(
            save_path,
            spike_times=np.array(spike_times),
            eeg_data=eeg_data,
            fs=fs,
            save_time=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        QMessageBox.information(None, "Save Successful", f"Spike data saved to:\n{save_path}")
        return True
    except Exception as e:
        QMessageBox.critical(None, "Save Failed", f"Error: {str(e)}")
        return False

def load_spike_data():
    load_path, _ = QFileDialog.getOpenFileName(
        None, "Load Spike Detection Results", "", "NPZ Files (*.npz);;All Files (*.*)"
    )
    if not load_path:
        return None
    try:
        data = np.load(load_path, allow_pickle=True)
        return {
            "spike_times": data["spike_times"].tolist(),
            "eeg_data": data["eeg_data"],
            "fs": data["fs"].item(),
            "save_time": data["save_time"].item()
        }
    except Exception as e:
        QMessageBox.critical(None, "Load Failed", f"Error: {str(e)}\nPlease confirm the file is a valid spike result file")
        return None