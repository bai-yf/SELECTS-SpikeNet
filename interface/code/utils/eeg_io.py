
"""EEG source resolution, reading, normalization, and interval reconstruction."""
import os
import re

import mne
import numpy as np

def resolve_eeg_file_path(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".fdt":
        set_path = os.path.splitext(file_path)[0] + ".set"
        if os.path.exists(set_path):
            return set_path
    return file_path


def read_supported_eeg(file_path, preload=False, verbose=False):
    file_path = resolve_eeg_file_path(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".set":
            return mne.io.read_raw_eeglab(file_path, preload=preload, verbose=verbose)
        if ext == ".edf":
            return mne.io.read_raw_edf(file_path, preload=preload, verbose=verbose)
    except Exception as e:
        if ext == ".set" and "Unknown mat file type" in str(e):
            raise ValueError(
                "This .set file is not a valid EEGLAB .set header readable by MNE, or the file is damaged/wrong.\n"
                "Please choose an EEGLAB-exported .set file, not .fdt, CSV, TXT, or another software's SET format.\n"
                f"Original error: {e}"
            ) from e
        raise
    raise ValueError(f"Only TXT, EDF, or EEGLAB SET files are supported. Current file: {os.path.basename(file_path)}")


def normalize_eeg_float32(eeg_data, invert=False):
    """Convert once to float32 and standardize each channel in place."""
    data = np.asarray(eeg_data, dtype=np.float32)
    if invert:
        np.negative(data, out=data)
    channel_mean = np.mean(data, axis=1, dtype=np.float64).astype(np.float32)[:, None]
    data -= channel_mean
    channel_std = np.std(data, axis=1, dtype=np.float64).astype(np.float32)[:, None]
    data /= channel_std + np.float32(1e-6)
    return data


def load_eeg_from_source(source_file_path, source_intervals, source_invert,
                          source_channel_names, expected_fs):
    """Rebuild normalized EEG data referenced by a lightweight result file."""
    raw = read_supported_eeg(source_file_path, preload=False, verbose=False)

    wanted_channels = [str(ch) for ch in source_channel_names]
    if wanted_channels:
        def norm_ch(name):
            text = re.sub(r'[^A-Z0-9]', '', str(name).upper())
            for token in ('EEG', 'REF', 'LE', 'AVG'):
                text = text.replace(token, '')
            return text

        available = {norm_ch(ch): ch for ch in raw.ch_names}
        picks = []
        for wanted in wanted_channels:
            wanted_norm = norm_ch(wanted)
            match = available.get(wanted_norm)
            if match is None:
                match = next((ch for ch in raw.ch_names if wanted_norm in norm_ch(ch)), None)
            if match is None:
                raise ValueError(f"EEG source is missing channel: {wanted}")
            picks.append(match)
        raw.pick_channels(picks, ordered=True)

    actual_fs = float(raw.info['sfreq'])
    if expected_fs and not np.isclose(actual_fs, float(expected_fs)):
        raise ValueError(
            f"EEG source sampling rate changed: result={expected_fs} Hz, source={actual_fs} Hz"
        )

    intervals = np.asarray(source_intervals, dtype=float).reshape(-1, 2)
    segment_specs = []
    if intervals.size:
        total_duration = raw.n_times / actual_fs
        for start_sec, end_sec in intervals:
            tmin = max(0.0, float(start_sec))
            tmax = min(total_duration, float(end_sec))
            if tmax > tmin:
                start_sample = max(0, int(round(tmin * actual_fs)))
                end_sample = min(raw.n_times, int(round(tmax * actual_fs)))
                if end_sample > start_sample:
                    segment_specs.append((start_sample, end_sample))
        if not segment_specs:
            raise ValueError("None of the saved EEG intervals exists in the source file.")
    else:
        segment_specs = [(0, raw.n_times)]

    total_samples = sum(end - start for start, end in segment_specs)
    eeg_data = np.empty((len(raw.ch_names), total_samples), dtype=np.float32)
    # Ten-minute chunks keep peak temporary memory modest while avoiding the
    # per-call overhead of hundreds of tiny reads from multi-gigabyte FDT files.
    chunk_samples = max(1, int(actual_fs * 600.0))
    write_pos = 0
    for start_sample, end_sample in segment_specs:
        for chunk_start in range(start_sample, end_sample, chunk_samples):
            chunk_end = min(end_sample, chunk_start + chunk_samples)
            block = raw.get_data(start=chunk_start, stop=chunk_end)
            block_len = chunk_end - chunk_start
            eeg_data[:, write_pos:write_pos + block_len] = block
            write_pos += block_len

    eeg_data = normalize_eeg_float32(eeg_data, invert=source_invert)
    return eeg_data, actual_fs
