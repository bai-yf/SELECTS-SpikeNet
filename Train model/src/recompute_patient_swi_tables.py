"""Recompute patient-level SWI tables from held-out window predictions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.metrics import compute_patient_swi_from_timestamps


REQUIRED_PREDICTION_COLUMNS = {
    'fold', 'patient_id', 'mid_time_sec', 'y_pred'
}
REQUIRED_ANNOTATION_COLUMNS = {'patient_id', 'spike_time_sec'}


def load_prediction_files(predictions_dir):
    files = sorted(Path(predictions_dir).glob('fold_*_predictions.csv'))
    if not files:
        raise FileNotFoundError(
            f'No fold_*_predictions.csv files found in {predictions_dir}'
        )
    frames = [pd.read_csv(path) for path in files]
    predictions = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f'Prediction files are missing columns: {sorted(missing)}')
    return predictions


def recompute_patient_swi(predictions, annotations, window_sec=0.2):
    missing = REQUIRED_ANNOTATION_COLUMNS - set(annotations.columns)
    if missing:
        raise ValueError(f'Annotation file is missing columns: {sorted(missing)}')

    annotation_groups = {
        str(patient_id): group['spike_time_sec'].to_numpy(dtype=float)
        for patient_id, group in annotations.groupby('patient_id', sort=False)
    }
    rows = []
    group_columns = ['fold', 'patient_id']
    for (fold, patient_id), patient_predictions in predictions.groupby(
            group_columns, sort=True):
        patient_id = str(patient_id)
        patient_predictions = patient_predictions.sort_values('mid_time_sec')
        mid_times = patient_predictions['mid_time_sec'].to_numpy(dtype=float)
        if len(mid_times) == 0:
            continue

        coverage_start = float(np.min(mid_times) - window_sec / 2)
        coverage_end = float(np.max(mid_times) + window_sec / 2)
        annotation_times = annotation_groups.get(patient_id, np.array([], dtype=float))
        metrics = compute_patient_swi_from_timestamps(
            annotation_times=annotation_times,
            window_mid_times=mid_times,
            window_predictions=patient_predictions['y_pred'].to_numpy(dtype=int),
            coverage_start_sec=coverage_start,
            coverage_end_sec=coverage_end,
        )
        rows.append({
            'fold': int(fold),
            'patient_id': patient_id,
            'n_windows': int(len(patient_predictions)),
            **metrics,
        })

    patient_table = pd.DataFrame(rows)
    if patient_table.empty:
        raise ValueError('No patient-level SWI rows were generated')
    summary = pd.DataFrame([{
        'n_patients': int(len(patient_table)),
        'mean_swi_true': float(patient_table['swi_true'].mean()),
        'mean_swi_pred': float(patient_table['swi_pred'].mean()),
        'swiae_mean': float(patient_table['swi_abs_error'].mean()),
        'swiae_std': float(patient_table['swi_abs_error'].std(ddof=1)),
        'swi_method': 'annotation_1s_epoch_midtime_prediction',
    }])
    return patient_table, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions-dir', required=True)
    parser.add_argument('--annotations-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--window-sec', type=float, default=0.2)
    args = parser.parse_args()

    predictions = load_prediction_files(args.predictions_dir)
    annotations = pd.read_csv(args.annotations_csv)
    patient_table, summary = recompute_patient_swi(
        predictions, annotations, window_sec=args.window_sec
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patient_table.to_csv(
        output_dir / 'patient_swi_per_fold.csv', index=False, encoding='utf-8-sig'
    )
    summary.to_csv(
        output_dir / 'patient_swi_summary.csv', index=False, encoding='utf-8-sig'
    )
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
