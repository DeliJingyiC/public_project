#!/bin/bash

# Training script for DLPO
# Remove SLURM-specific configurations for public release

# Uncomment and modify these lines for SLURM clusters:
# #SBATCH --time=05:30:00
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH -p gpu
# #SBATCH --output=output/%j.log
# #SBATCH --mail-type=FAIL
# #SBATCH --ntasks-per-node=2
# #SBATCH --gpus-per-node=2
# #SBATCH --gpu_cmode=exclusive
# #SBATCH --nodes=2
# #SBATCH --mem=128G

# Load CUDA if available (modify path as needed)
# module load cuda/11.8.0

# Activate your conda environment
# source .ascendenv/bin/activate

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create output directory
mkdir -p output
WORK_DIR=output/$(date +%Y%m%d_%H%M%S)

# Clean buffer directory
rm -rf buffer1/*

# Run training
python -m trainer \
    --save_audio_dir=$WORK_DIR \
    --resume_from=1059 \
    --loss_type=dlpo5 \
    --mode predict_file \
    --pretrained_model weights/nisqa.tar


