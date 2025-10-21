#!/usr/bin/env bash
set -e

# ---------------------------------------
# NOBLE Installation Script
# Run this from within the 'noble' folder
# ---------------------------------------

echo "=========================================="
echo "Installing NOBLE Environment"
echo "=========================================="

# --- Step 1: Check Python version ---
echo "[1/6] Checking Python version..."
if ! command -v python &> /dev/null; then
    echo "Python not found. Please install Python 3.10 or higher."
    exit 1
fi

PYVER=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$(printf '%s\n' "3.10" "$PYVER" | sort -V | head -n1)" != "3.10" ]]; then
    echo "Python version $PYVER detected. Python 3.10 or higher is required."
    exit 1
fi
echo "Python version $PYVER OK."
echo ""

# --- Step 2: Check for conda ---
echo "[2/6] Checking Conda..."
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda or Anaconda first. For linux, find more information here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html"
    exit 1
fi
echo "Conda found."
echo ""

# --- Step 3: Create and activate Conda environment ---
echo "[3/6] Creating conda environment..."
conda env create -f environment.yml || echo "Environment may already exist."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate noble
echo ""

# --- Step 4: Install NeuralOperator ---
echo "[4/6] Installing NeuralOperator..."
cd ..
if [ ! -d "neuraloperator" ]; then
    git clone https://github.com/neuraloperator/neuraloperator
else
    echo "neuraloperator/ already exists, skipping clone."
fi
cd neuraloperator
pip install -r requirements.txt
pip install -e .
echo ""

# --- Step 5: Check CUDA and compute capability ---
echo "[5/6] Checking CUDA and GPU capability..."
if python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    echo "CUDA available: version $CUDA_VERSION"

    CC_MAJOR=$(python -c "import torch; print(torch.cuda.get_device_capability(0)[0])")
    echo "Detected GPU compute capability: ${CC_MAJOR}"

    if (( CC_MAJOR <= 6 )); then
        echo "Detected compute capability ≤ 6.0 — installing compatible torch version."
        pip install torch==2.6.0
    fi
else
    echo "CUDA not available — continuing with current torch installation."
fi
echo ""

# --- Step 6: Install NOBLE packages ---
echo "[6/6] Installing NOBLE packages..."
cd ../noble/src
pip install -e .
cd ../inference
pip install -e .
echo ""

echo "=========================================="
echo "NOBLE installation complete!"
echo "To activate the environment: conda activate noble"
echo "=========================================="
