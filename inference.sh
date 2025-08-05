#!/bin/bash

# Inference script for DLPO
# Remove SLURM-specific configurations for public release

# Uncomment and modify these lines for SLURM clusters:
# #SBATCH --time=1:00:00
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH -p gpu
# #SBATCH --output=output/%j.log
# #SBATCH --mail-type=FAIL
# #SBATCH --ntasks-per-node=2
# #SBATCH --gpus-per-node=2
# #SBATCH --gpu_cmode=exclusive
# #SBATCH --nodes=2

# Load CUDA if available (modify path as needed)
# module load cuda/11.8.0

# Activate your conda environment
# source .ascendenv/bin/activate

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create output directory
mkdir -p output
WORK_DIR=output/$(date +%Y%m%d_%H%M%S)

# Check Python version
python --version

# Run inference
python inference.py \
    --checkpoint="checkpoint/model_checkpoint.ckpt" \
    --save_audio_dir=$WORK_DIR \
    --resume_from=1059 \
    --filename='DLPO'

# Example commands for different models (uncomment as needed):
# python inference.py --checkpoint="checkpoint/klinr_model.ckpt" --save_audio_dir=$WORK_DIR --resume_from=1059 --filename='klinR'
# python inference.py --checkpoint="checkpoint/baseline_model.ckpt" --save_audio_dir=$WORK_DIR --resume_from=1755
