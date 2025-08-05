# DLPO: Diffusion-based Language Policy Optimization for Text-to-Speech

This repository contains the implementation of DLPO (Diffusion-based Language Policy Optimization), a reinforcement learning approach for improving text-to-speech synthesis quality using diffusion models.

## Overview

DLPO combines diffusion models with reinforcement learning to optimize TTS synthesis quality. The approach uses:
- WaveGrad2 as the base diffusion model
- NISQA for audio quality assessment
- UTMOS for speech quality evaluation

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd DLPO
```

2. Follow the installation guide:
```bash
# Quick setup (recommended)
bash scripts/setup_environment.sh

# Or see INSTALL.md for detailed instructions
```

3. Activate the conda environment:
```bash
conda activate dlpo
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## External Dependencies

This project requires several external modules that need to be installed separately:

### NISQA
Download and install NISQA for audio quality assessment:
```bash
git clone https://github.com/gabrielmittag/NISQA.git
cd NISQA
pip install -e .
```

### UTMOS
Download and install UTMOS for speech quality evaluation:
```bash
git clone https://github.com/sarulab-speech/UTMOS.git
cd UTMOS
pip install -e .
```

## Configuration

1. Update the configuration file `hparameter.yaml`:
   - Set your dataset paths in the `data` section
   - Adjust model parameters as needed
   - Configure training parameters

2. Create necessary directories:
```bash
mkdir -p data/LJSpeech/preprocessed
mkdir -p checkpoint
mkdir -p tensorboard
mkdir -p test_sample/result
mkdir -p buffer1
```

## Usage

### Training

To train the model:

```bash
# For SLURM clusters
sbatch train.sh

# For local training
python trainer.py --config hparameter.yaml
```

### Inference

To run inference:

```bash
# For SLURM clusters
sbatch inference.sh

# For local inference
python inference.py --checkpoint path/to/checkpoint.ckpt --text "Hello world"
```

### Configuration Parameters

Key parameters in `hparameter.yaml`:

- `data.train_dir`: Path to training data directory
- `data.val_dir`: Path to validation data directory
- `train.batch_size`: Training batch size
- `train.adam.lr`: Learning rate
- `wavegrad.is_large`: Use large or base model architecture
- `ddpm.max_step`: Number of diffusion steps
- `ddpm.infer_step`: Number of inference steps

## Data Format

The model expects:
- Audio files in WAV format (22050 Hz sampling rate)
- Text transcriptions in a metadata file
- Preprocessed mel-spectrograms

## Model Architecture

- **Text Encoder**: Convolutional encoder for text processing
- **WaveGrad Decoder**: Diffusion-based audio generation
- **Speaker Embedding**: Speaker-specific embeddings
- **Quality Assessment**: Integration with NISQA and UTMOS

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{dlpo2025,
  title={Fine-Tuning Text-to-Speech Diffusion Models Using Reinforcement Learning with Human Feedback},
  author={Jingyi Chen, Ju Seung Byun, Micha Elsner, Pichao Wang, Andrew Perrault
},
  journal={Interspeech},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- WaveGrad2 implementation adapted from [ivanvovk/WaveGrad](https://github.com/ivanvovk/WaveGrad)
- NISQA for audio quality assessment
- UTMOS for speech quality evaluation 