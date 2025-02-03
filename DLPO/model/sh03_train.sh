#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=PAS1957
#SBATCH --output=output/%j.log
#SBATCH --mail-type=FAIL
#SBATCH --ntasks=48
#SBATCH --gpus-per-node=2
#SBATCH --gpu_cmode=exclusive
#SBATCH -p gpuserial-48core
#SBATCH -N 1

source .workenv/bin/activate

set -ex

if test -z $SLURM_JOB_ID; then
    export SLURM_JOB_ID=$(date +%s)
    echo "then $SLURM_JOB_ID"
fi
mkdir -p output/$SLURM_JOB_ID
WORK_DIR=output/$SLURM_JOB_ID

python -m trainer --save_audio_dir=$WORK_DIR

# salloc -t 1:00:00 \
#     -N 1 \
#     -p gpudebug-48core \
#     -A PAS1957 \
#     --ntasks-per-node=28 \
#     --gpus-per-node=2 \
#     --gpu_cmode=exclusive \
#     srun \
#     --pty /bin/bash

# salloc -t 2:30:00 \
#     -N 1 \
#     -p gpuserial-48core \
#     -A PAS1957 \
#     --ntasks-per-node=28 \
#     --gpus-per-node=2 \
#     --gpu_cmode=exclusive \
#     srun \
#     --pty /bin/bash

# module load cuda/11.8.0
# nvidia-smi
# watch -n 1 nvidia-smi
# squeue -u delijingyic --start
# sinfo -p gpuserial-48core
# scancel -u delijingyic
