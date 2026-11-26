
## Project Overview
Deep learning-based (ResNet+SPP+KAN) automatic detection system for Benign Epilepsy with Centrotemporal Spikes (BECT/SELECT) in EEG signals.

## Model Architecture
1.Feature Extraction: Raw signal + first/second order differences + SNE features
2.Backbone Network: Lightweight ResNet-34 (`[3, 4, 6, 3]` residual blocks)
3.Spatial Pyramid Pooling: Multi-scale feature aggregation
4.Classifier: KAN (Kolmogorov-Arnold Network)

## Final Data Augmentation (AUG06)
Each positive training segment receives one augmented copy. The copy first receives Gaussian noise with a standard deviation equal to 3% of the segment standard deviation, then is always reversed along the true time axis. Frequency-domain perturbation is not part of the final protocol.

## Project Structure
Train model/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   └── cache/
│       ├── dataset_windows_0p2s_500hz_44person.npz
│       └── dataset_windows_0p2s_500hz_44person_metadata.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── light_resnet.py
│   │   ├── kan_classifier.py
│   │   └── fusion_model.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── augmentation.py
│   │   └── metrics.py
│   └── train.py
└── results/                   
    ├── fold_1_best_weights.pth
    ├── fold_2_best_weights.pth
    ├── ... (another.pth file)
    └── training_logs.csv

## Configuration Modification
Modify the following paths in config/config.yaml:
```yaml
data:
  cache_npz_path: "data/cache/dataset_windows_0p2s_500hz_44person.npz"
  cache_metadata_path: "data/cache/dataset_windows_0p2s_500hz_44person_metadata.csv"
  results_dir: "results"

device: "cuda:0"  # Modify based on your GPU configuration

```

## Installation and Setup
1.Prepare project structure:
```bash
mkdir -p BECT_EEG_Analysis/{config,data/{raw,metadata},src/{models,utils},results}
```
2.Place the code files in corresponding directories
3.Prepare the preprocessed window cache and its patient/window metadata CSV in data/cache/
4.Install dependencies:
```bash
cd BECT_EEG_Analysis
pip install -r requirements.txt
```
5.Install KAN library (if not already installed):
```bash
git clone https://github.com/Blealtan/efficient-kan.git
cd efficient-kan
pip install -e .
```
6.Run training:
```bash
python src/train.py
```
7.Output Results:
After training, the following files are generated in results/:
    Best weights for each fold (fold_X_best_weights.pth)
    Cross-validation summary results (cross_validation_summary.csv)
