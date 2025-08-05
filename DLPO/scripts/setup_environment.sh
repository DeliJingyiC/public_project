#!/bin/bash

# Setup script for DLPO environment
# This script helps set up the environment and install dependencies

set -e

echo "Setting up DLPO environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
if [ -f "../environment.yml" ]; then
    echo "Using environment.yml file..."
    conda env create -f ../environment.yml
else
    echo "Creating environment manually..."
    conda create -n dlpo python=3.9 -y
    
    # Activate conda environment
    echo "Activating conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate dlpo
    
    # Install PyTorch with CUDA support (adjust CUDA version as needed)
    echo "Installing PyTorch..."
    conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip install -r ../requirements.txt
fi

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dlpo

# Create necessary directories
echo "Creating directories..."
mkdir -p data/LJSpeech/preprocessed
mkdir -p checkpoint
mkdir -p tensorboard
mkdir -p test_sample/result
mkdir -p buffer1
mkdir -p output
mkdir -p mos_results
mkdir -p weights

# Clone external dependencies
echo "Setting up external dependencies..."

# NISQA
if [ ! -d "NISQA" ]; then
    echo "Cloning NISQA..."
    git clone https://github.com/gabrielmittag/NISQA.git
    cd NISQA
    pip install -e .
    cd ..
else
    echo "NISQA already exists, skipping..."
fi

# UTMOS
if [ ! -d "UTMOS" ]; then
    echo "Cloning UTMOS..."
    git clone https://github.com/sarulab-speech/UTMOS.git
    cd UTMOS
    pip install -e .
    cd ..
else
    echo "UTMOS already exists, skipping..."
fi

# Download lexicon if not exists
if [ ! -f "lexicon/librispeech-lexicon.txt" ]; then
    echo "Creating lexicon directory..."
    mkdir -p lexicon
    echo "Please download librispeech-lexicon.txt and place it in the lexicon/ directory"
    echo "You can find it at: https://github.com/keithito/tacotron/blob/master/data/cmu_dictionary"
fi

echo "Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "conda activate dlpo"
echo ""
echo "Next steps:"
echo "1. Update paths in hparameter.yaml to point to your dataset"
echo "2. Download and place model checkpoints in the appropriate directories"
echo "3. Prepare your dataset according to the format described in README.md"
echo "4. Run training: bash train.sh"
echo "5. Run inference: bash inference.sh" 