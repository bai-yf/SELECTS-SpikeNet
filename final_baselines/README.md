# Final baseline implementations

This directory contains training/evaluation code for the five baseline models
used in the manuscript comparison. Baseline inference interfaces and trained
baseline checkpoints are intentionally not distributed. The public clinical
interface and inference checkpoint correspond only to the proposed model.

| Manuscript baseline | Training entry point |
| --- | --- |
| EEGPT | `EEGPT/train.py` |
| EEGNeX | `EEGNeX/train.py` |
| EEGNet | `EEGNet/train.py` |
| EEGFormer | `EEGFormer/train.py` |
| LaBraM | `LaBraM/train.py` |

The scripts are exported from the experiment notebooks without notebook output
or execution metadata. Runtime paths can be set with `SELECTS_DATA_DIR`,
`SELECTS_META_PATH`, `SELECTS_CACHE_NPZ`, `SELECTS_CACHE_METADATA`, and
`SELECTS_RESULT_DIR`. GPU selection follows `CUDA_VISIBLE_DEVICES`.

## Shared experiment configuration

Only the model architecture and the model-specific input adapter differ between
baselines. All five scripts use the same experiment configuration as the
proposed model:

- patient-grouped 5-fold `GroupKFold`, seed 42;
- 16-channel, 500 Hz, 0.2-second windows;
- one additional sample for every positive training window only;
- 3% Gaussian noise followed by reversal of the true time axis;
- batch size 64, 50 epochs, learning rate `1e-4`, weight decay `1e-4`;
- focal loss with alpha 0.65 and gamma 2.0; and
- early stopping patience 5 with minimum delta 0.003.

No augmentation is applied to validation or held-out prediction windows.

## Data and weights

Patient EEG data, annotations, cached windows, result tables, build artifacts,
and trained baseline fold checkpoints are not included. Each `train.py` writes
its own fold checkpoints and held-out predictions to its configured output
directory.

## Upstream pretrained checkpoints

EEGPT uses the official
[`eegpt_mcae_58chs_4s_large4E.ckpt`](https://figshare.com/s/e37df4f8a907a866df4b)
checkpoint from the [EEGPT repository](https://github.com/BINE022/EEGPT).
Clone EEGPT and place the file at
`downstream/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt`, or set
`EEGPT_REPO_PATH` and `EEGPT_PRETRAINED_CHECKPOINT`.

LaBraM uses the official `labram-base.pth` checkpoint and source from the
[LaBraM repository](https://github.com/935963004/LaBraM). Set
`LABRAM_REPO_PATH` and `LABRAM_PRETRAINED_CHECKPOINT` if they are stored
elsewhere.

Run a baseline from its directory with:

```bash
python train.py
```

## Final patient-level SWI

The training scripts save held-out window predictions but do not approximate
SWI by grouping every five array rows. Recompute the manuscript SWI/SWIAE from
original annotation timestamps and window `mid_time_sec` values with:

```bash
python "Train model/src/recompute_patient_swi_tables.py" \
  --predictions-dir final_baselines/<MODEL>/outputs/predictions \
  --annotations-csv /path/to/spike_annotations.csv \
  --output-dir final_baselines/<MODEL>/outputs \
  --window-sec 0.2
```

This uses the manuscript method identifier
`annotation_1s_epoch_midtime_prediction` and reports the macro-average of the
patient-level absolute SWI errors.

The older article-reproduction scripts are archived separately in
[`../legacy_baseline_reproductions`](../legacy_baseline_reproductions) and are
not final manuscript baselines.
