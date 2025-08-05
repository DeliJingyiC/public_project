# Installation Guide

This guide provides multiple ways to set up the DLPO environment.

## Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended)
- Anaconda or Miniconda

## Method 1: Automated Setup (Recommended)

Run the setup script which will create a conda environment and install all dependencies:

```bash
bash scripts/setup_environment.sh
```

## Method 2: Manual Conda Environment Setup

1. Create conda environment from environment file:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate dlpo
```

3. Create necessary directories:
```bash
mkdir -p data/LJSpeech/preprocessed
mkdir -p checkpoint
mkdir -p tensorboard
mkdir -p test_sample/result
mkdir -p buffer1
mkdir -p output
mkdir -p mos_results
mkdir -p weights
```

4. Install external dependencies:
```bash
# NISQA
git clone https://github.com/gabrielmittag/NISQA.git
cd NISQA
pip install -e .
cd ..

# UTMOS
git clone https://github.com/sarulab-speech/UTMOS.git
cd UTMOS
pip install -e .
cd ..
```

## Method 3: Manual Installation

1. Create conda environment:
```bash
conda create -n dlpo python=3.9 -y
conda activate dlpo
```

2. Install PyTorch with CUDA:
```bash
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Follow steps 3-4 from Method 2.

## External Dependencies

### NISQA
For audio quality assessment:
```bash
git clone https://github.com/gabrielmittag/NISQA.git
cd NISQA
pip install -e .
cd ..
```

### UTMOS
For speech quality evaluation:
```bash
git clone https://github.com/sarulab-speech/UTMOS.git
cd UTMOS
pip install -e .
cd ..
```

### Lexicon
Download the LibriSpeech lexicon:
```bash
mkdir -p lexicon
# Download librispeech-lexicon.txt from:
# https://github.com/keithito/tacotron/blob/master/data/cmu_dictionary
```

## Verification

To verify the installation:

```bash
conda activate dlpo
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import pytorch_lightning; print('PyTorch Lightning installed')"
```

## Troubleshooting

### CUDA Issues
- Make sure you have the correct CUDA version installed
- Update the CUDA version in the installation commands if needed
- Check GPU compatibility with `nvidia-smi`

### Import Errors
- Ensure the conda environment is activated: `conda activate dlpo`
- Reinstall dependencies if needed: `pip install -r requirements.txt`

### External Dependencies
- Make sure NISQA and UTMOS are properly installed
- Check that the lexicon file is in the correct location

## Next Steps

After installation:

1. Update paths in `hparameter.yaml`
2. Prepare your dataset
3. Download model checkpoints
4. Start training or inference 