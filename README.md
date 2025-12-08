SELECTS-SpikeNet
《A Clinical Software for End-to-End Automatic Spike Detection in SeLECTS from Long-Term EEG Based on Multi-Domain Feature Fusion》

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://riverbankcomputing.com/software/pyqt/)
[![MATLAB](https://img.shields.io/badge/Integration-MATLAB-orange.svg)](https://www.mathworks.com/)
[![EEGLAB](https://img.shields.io/badge/Toolbox-EEGLAB-yellow.svg)](https://sccn.ucsd.edu/eeglab/)
An integrated clinical software system for end-to-end automatic spike detection in Self-Limited Epilepsy with Centrotemporal Spikes (SeLECTS) from long-term EEG recordings. The system combines signal processing, machine learning, and interactive visualization for comprehensive EEG analysis.

https://docs/screenshot.png

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

5.the Kolmogorov-Arnold Network (KAN)  from the efficient-kan repository.（reference:https://github.com/Blealtan/efficient-kan）
```text
SELECTS-SpikeNet/
├── main.py                      # Main application entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── models/                      # Model architecture and weights
│   ├── __init__.py
│   ├── kan_resnet_model.py     # KAN+ResNet hybrid model
│   └── weights/
│       └── fold_3_best_weights.pth  # Pretrained model
├── data/                        # Data directories
│   ├── txt/                    # Raw TXT format EEG data
│   ├── set/                    # SET format EEG data
│   ├── filtered/               # Preprocessed SET files
│   ├── sleep_csv/              # Sleep staging CSV files
│   ├── temp_set/               # Temporary SET files
│   └── result/                 # Spike detection results
├── docs/                        # Documentation and screenshots
└── utils/                       # Utility functions
    ├── preprocessing.py        # MATLAB preprocessing functions
    ├── feature_extraction.py   # Multi-domain feature extraction
    └── visualization.py        # Plotting and visualization utilities
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






```text

```



```bash

```
