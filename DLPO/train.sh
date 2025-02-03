#!/bin/bash
#SBATCH --time=05:30:00
#SBATCH --account=PAS2138
#SBATCH -p gpu
#SBATCH --output=output/%j.log
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --gpu_cmode=exclusive
#SBATCH --nodes=2
#SBATCH --mem=128G


module load cuda/11.8.0
# module load nccl
# module load cudnn/8.6.0.163-11.8

# # #----------------------------
# # # this is the part for the new torch environment condition, comment or uncomment it as you need
# CONDA_ENV="wavgrd2-torch"
# source $MINICONDA/etc/profile.d/conda.sh
# conda activate $CONDA_ENV
# # # ----------------------------

source .ascendenv/bin/activate
echo $SLURM_JOB_ID
echo $SLURM_PROCID
echo $SLURM_LOCALID
echo $SLURM_NODEID
echo $SLURM_NTASKS

set -ex
# add direct cuda env mapping, may fix cuda linking issue
# export CUDA_HOME=/apps/spack/0.17/root/linux-rhel8-zen/cuda/gcc/8.4.1/11.8.0-eyqbbsj
# export CUDA_ROOT=$CUDA_HOME
# export LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBRARY_PATH

# if test -z $SLURM_JOB_ID; then
#     export SLURM_JOB_ID=$(date +%s)
#     echo "then $SLURM_JOB_ID"
# fi
mkdir -p output/$SLURM_JOB_ID
WORK_DIR=output/$SLURM_JOB_ID

rm -rf /users/PAS2062/delijingyic/project/wavegrad2/buffer1/*

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
srun python -m trainer --save_audio_dir=$WORK_DIR --resume_from=1059 --loss_type=dlpo5 --mode predict_file --pretrained_model weights/nisqa.tar


