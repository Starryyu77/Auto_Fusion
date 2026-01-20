#!/bin/bash
# setup_m4_track_a.sh
# Description: Sets up the Conda environment for MM-CoT on Apple Silicon (M4).
# Usage: source setup_m4_track_a.sh

ENV_NAME="mmcot_mps"
PYTHON_VERSION="3.10" # 3.10 is stable for most ML libraries

echo "=================================================="
echo "   Setting up Environment: $ENV_NAME (Apple Silicon)   "
echo "=================================================="

# Check for Conda
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "Error: Conda/Mamba is not installed. Please install Miniforge or Anaconda first."
    exit 1
fi

echo "Using: $CONDA_CMD"

# Create Environment
echo "Creating environment $ENV_NAME with Python $PYTHON_VERSION..."
$CONDA_CMD create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate Environment
# Note: 'conda activate' inside script might require 'eval' hook
eval "$($CONDA_CMD shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch (MPS Support)
echo "Installing PyTorch with MPS support..."
$CONDA_CMD install pytorch::pytorch torchvision torchaudio -c pytorch -y

# Install Core Dependencies
echo "Installing Transformers, Accelerate, and others..."
pip install transformers accelerate sentencepiece datasets psutil pylint black protobuf

echo "=================================================="
echo "   Setup Complete! Activate with: conda activate $ENV_NAME"
echo "=================================================="
