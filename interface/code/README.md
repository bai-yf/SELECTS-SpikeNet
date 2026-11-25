# EEG spike-detection interface

Run `main.py` from this directory, or double-click `run_V9_interface.bat` when using the local Conda environment named `interface`.

The interface loads `fold_3_best_weights.pth` and the bundled `efficient-kan/src` directory relative to this folder. Preprocessed data and temporary EEGLAB files are written under `interface/data`.

The current interface supports TXT, EDF, and EEGLAB SET input; sleep-stage segment import; responsive MATLAB preprocessing; batched inference; lightweight result files; manual spike correction with autosave; long-term review timing; keyboard and mouse navigation; amplitude scaling; and spatial-distribution visualization.

## Weight provenance

The project owner designates the checked-in `fold_3_best_weights.pth` as the final AUG06 inference weight (Gaussian noise plus true time reversal). The interface loads this file directly and does not download or substitute another weight.

- Size: 39935594 bytes
- SHA-256: `C097AD93595246434172342223AC2469496DDD2B18E3C5DB757305156AD04595`