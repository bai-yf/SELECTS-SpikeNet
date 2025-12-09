SELECTS-SpikeNet
《A Clinical Software for End-to-End Automatic Spike Detection in SeLECTS from Long-Term EEG Based on Multi-Domain Feature Fusion》

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://riverbankcomputing.com/software/pyqt/)
[![MATLAB](https://img.shields.io/badge/Integration-MATLAB-orange.svg)](https://www.mathworks.com/)
[![EEGLAB](https://img.shields.io/badge/Toolbox-EEGLAB-yellow.svg)](https://sccn.ucsd.edu/eeglab/)

An integrated clinical software system for end-to-end automatic spike detection in Self-Limited Epilepsy with Centrotemporal Spikes (SeLECTS) from long-term EEG recordings. The system combines signal processing, machine learning, and interactive visualization for comprehensive EEG analysis.

[https://github.com/bai-yf/SELECTS-SpikeNet/blob/master/interface/interface.png]

🚀 Quick Start
1. Environment Setup
Option A: Using Conda (Recommended)
```bash
# Create conda environment
conda create -n selects-spikenet python=3.9
conda activate selects-spikenet

# Install core dependencies
pip install -r requirements.txt
```
Option B: Using Virtual Environment
```bash
# Create virtual environment
python -m venv eeg_env

# Activate on Windows
eeg_env\Scripts\activate

# Activate on Linux/Mac
source eeg_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

2. MATLAB Engine Installation
```bash
# Navigate to MATLAB engine directory
cd "C:\Program Files\MATLAB\R2024a\extern\engines\python"

# Install MATLAB engine for Python
python setup.py install

# Alternative: Install from pip (if available)
pip install matlabengine
```

3. EEGLAB Setup
Download EEGLAB from https://sccn.ucsd.edu/eeglab/download.php
Extract to: D:\MATLAB\R2024a\toolbox\eeglab (or your preferred path)
Add to MATLAB path:
```bash
addpath(genpath('D:\MATLAB\R2024a\toolbox\eeglab'));
savepath;
```

4. Model Weights
Fold_3_best_weights.pth is the pretrained model weight file; alternatively, you can use the code in the model documents to train the model on your own.

5.Run the Interface
Note: The interface runs locally.
Navigate to the target directory via your local command line:
```bash
cd SELECTS-SpikeNet/interface/code
```
Launch the interface with the following command:
```bash
python main.py
```

6.Model training is executed on a GPU-equipped server.
6.the Kolmogorov-Arnold Network (KAN)  from the efficient-kan repository.（reference:https://github.com/Blealtan/efficient-kan）


📁 File Directory Structure
```text
SELECTS-SpikeNet/
├── interface/                          # GUI application (for users)
│   └── code/
│       ├── main.py                     # Main GUI application entry
│       ├── config.py                   # GUI configuration settings
│       ├── requirements.txt            # GUI dependencies
│       ├── fold_3_best_weights.pth     # Pre-trained model for inference
│       ├── efficient-kan/              # KAN implementation
│       ├── core/                       # Core processing modules
│       │   ├── __init__.py
│       │   ├── models.py              # Model loading and inference
│       │   ├── features.py            # Feature extraction for inference
│       │   ├── preprocessing.py       # Real-time EEG preprocessing
│       │   └── metrics.py             # Real-time metrics calculation
│       ├── ui/                         # User interface components
│       │   ├── __init__.py
│       │   └── dialogs.py             # GUI dialog windows
│       ├── visualization/              # Result visualization
│       │   ├── __init__.py
│       │   └── brain_topography.py    # Brain map plotting
│       ├── utils/                      # GUI utilities
│       │   ├── __init__.py
│       │   └── file_io.py             # File handling for GUI
│   └── data/                          # GUI application data directory
│       ├── txt/                       # Raw TXT EEG files
│       ├── set/                       # SET format EEG files
│       ├── filtered/                  # Filtered/preprocessed SET files
│       ├── sleep_csv/                 # Sleep stage annotation files
│       ├── temp_set/                  # Temporary processing files
│       └── result/                    # Detection results
│
└── Train model/                       # Training code (for developers/researchers)
    ├── README.md                      # Training instructions
    ├── requirements.txt               # Training dependencies
    ├── config/                        # Training configuration
    │   └── config.yaml                # YAML config file
    ├── data/                          # Training data
    │   ├── raw/                       # Raw training EEG data
    │   │   ├── subject1.set
    │   │   ├── subject2.set
    │   │   └── ...
    │   └── metadata/                  # Training labels
    │       └── spick_time_label.xlsx  # Column1: patient ID, Column2+: spike times
    ├── src/                           # Training source code
    │   ├── __init__.py
    │   ├── data_loader.py             # Data loading and preprocessing
    │   ├── models/                    # Model definitions
    │   │   ├── __init__.py
    │   │   ├── light_resnet.py       # ResNet backbone
    │   │   ├── kan_classifier.py     # KAN classifier
    │   │   └── fusion_model.py       # Combined model
    │   ├── utils/                     # Training utilities
    │   │   ├── __init__.py
    │   │   ├── preprocessing.py      # Training preprocessing
    │   │   ├── augmentation.py       # Data augmentation
    │   │   └── metrics.py            # Training metrics
    │   └── train.py                   # Main training script
    └── results/                       # Training outputs
        ├── fold_1_best_weights.pth   # Cross-validation model 1
        ├── fold_2_best_weights.pth   # Cross-validation model 2
        ├── fold_3_best_weights.pth   # Cross-validation model 3
        ├── ...                        # Additional folds
        └── training_logs.csv         # Training logs
 
```


🧠 Model Architecture
Hybrid KAN+ResNet Model
```text
Input (16 channels × 200ms window)
    │
    ▼
[Feature Extraction]
    ├── Temporal derivatives (x0, x1, x2)
    ├── SNE (Spike Nonlinear Energy) features
    └── Multi-channel fusion
    │
    ▼
[LightResNet-EEG Backbone]
    ├── Conv2D + BatchNorm
    ├── BasicBlock × 4 layers
    ├── Spatial Pyramid Pooling
    └── 128-dim feature vector
    │
    ▼
[KAN Classifier]
    ├── Kolmogorov-Arnold Network layers
    ├── 128 → 64 → 2 architecture
    └── Probability output
    │
    ▼
Output: Spike (1) / Non-spike (0) with confidence
```

🛠️ Development
Building from Source
```bash
# Clone repository
git clone https://github.com/yourusername/SELECTS-SpikeNet.git
cd SELECTS-SpikeNet

# Install in development mode
pip install -e .

# Run tests
python main.py/
```

📝 Citation
If you use this software in your research, please cite:
```bibtex
@software{selects_spikenet_2024,
  title = {SELECTS-SpikeNet: A Clinical Software for End-to-End Automatic Spike Detection in SeLECTS},
  author = {Your Name and Collaborators},
  year = {2024},
  url = {https://github.com/yourusername/SELECTS-SpikeNet}
}
```
