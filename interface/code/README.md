# EEG spike-detection interface

Run `main.py` from this directory, or double-click `run_V9_interface.bat` when using the local Conda environment named `interface`.

The interface loads `fold_3_best_weights.pth` and the bundled `efficient-kan/src` directory relative to this folder. Preprocessed data and temporary EEGLAB files are written under `interface/data`.

The current interface supports TXT, EDF, and EEGLAB SET input; sleep-stage segment import; responsive MATLAB preprocessing; batched inference; lightweight result files; manual spike correction with autosave; long-term review timing; keyboard and mouse navigation; amplitude scaling; and spatial-distribution visualization.

## Weight provenance

The checked-in weight must be the fold selected from the final AUG06 training run (Gaussian noise plus true time reversal). Before publishing a new weight, replace `fold_3_best_weights.pth` and record its source run and SHA-256 checksum in the release notes. The code does not silently download or substitute a weight.