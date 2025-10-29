# $\texttt{NOBLE}$: Neural Operator for Biological Learning and Electrophysiology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

$\texttt{NOBLE}$ is a neural operator-based framework for modeling and predicting electrophysiological responses in biological neurons. The project leverages Fourier Neural Operators (FNO) to learn complex mappings between input stimuli and neuronal responses, enabling efficient simulation and analysis of neural dynamics. This repository accompanies our [NeurIPS 2025 paper](https://arxiv.org/abs/2506.04536).

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the `noble` repository:
```bash
git clone https://github.com/neuraloperator/noble.git
cd noble
```

2. Run the installation script to install the `nerualoperator` library (outside the `noble` directory) and the `noble` codebase:
```
bash install_noble.sh
```

## Project Structure

```
noble/
├── src/                          # Main source code
│   └── training/                 # Training pipeline and models
│       ├── engine/               # Training engines
│       ├── data/                 # Data loading utilities
│       ├── neuro/                # Neuroscience-specific modules
│       └── configs/              # Configuration files
│
├── inference/                    # Example scripts and utilities
│   ├── ensemble_generation.py
│   ├── arbitrary_ensemble_generation.py
│   ├── compute_ephys_features.py
│   ├── compare_ephys_features_experiments.py
│   ├── generate_FI_curve.py
│   ├── noble_models/             # Pre-trained models for inference
│   └── utils/                    # Utility functions for inference scripts
│
├── data/                         # Data storage
│   └── e-features/               # Electrophysiological features
│
└── training_scripts/             # SLURM job submission scripts
```

## Features

- **Fourier Neural Operator (FNO) Architecture**: Advanced neural operator for learning complex temporal dynamics
- **Electrophysiological Feature Extraction**: Automated computation of spike features using eFEL
- **Multi-scale Modeling**: Support for different temporal resolutions and downsampling factors
- **Embedding Systems**: Flexible embedding of amplitude, frequency, and electrophysiological features
- **Biophysical Neuron Models**: Integration with detailed biophysical simulations
- **Training and Fine-tuning**: Comprehensive training pipeline with configurable hyperparameters
- **Visualization Tools**: Built-in plotting and analysis utilities

## Test Examples
We provide five different example scripts in the `noble/inference/` directory to help you explore and test the trained $\texttt{NOBLE}$ model on:
- Generating ensemble predictions
- Computing electrophysiological features
- Generating frequency-current curves

```bash
cd noble/inference
```

## Citation
If you find this work useful, please cite:

```bibtex
@inproceedings{ghafourpour2025noble,
  title     = {NOBLE: Neural Operator with Biologically-informed Latent Embeddings to Capture Experimental Variability in Biological Neuron Models},
  author    = {Ghafourpour, Luca and Duruisseaux, Valentin and Tolooshams, Bahareh and Wong, Philip H and Anastassiou, Costas A and Anandkumar, Anima},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {39},
  year      = {2025},
  doi       = {arXiv:2506.04536}
}
```

## Contact

For questions and support, please contact Luca Ghafourpour (ldg34@cam.ac.uk).
