#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=PAS1957
#SBATCH -p gpu
#SBATCH --output=output/%j.log
#SBATCH --mail-type=FAIL
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --gpu_cmode=exclusive
#SBATCH --nodes=2

# module load cuda/11.8.0

source .ascendenv/bin/activate
echo $SLURM_JOB_ID
echo $SLURM_PROCID
echo $SLURM_LOCALID
echo $SLURM_NODEID
echo $SLURM_NTASKS
set -ex
set -ex

if test -z $SLURM_JOB_ID; then
    export SLURM_JOB_ID=$(date +%s)
    echo "then $SLURM_JOB_ID"
fi
mkdir -p output/$SLURM_JOB_ID
WORK_DIR=output/$SLURM_JOB_ID

python --version
python sh04_inference_finetune.py --checkpoint="/users/PAS2062/delijingyic/project/wavegrad2/checkpoint/wavegrad2_10_01_14_dlpo_epoch=6_loss=-3.660708427429199test.ckpt" --save_audio_dir=$WORK_DIR --resume_from=1059 --filename='DLPO'

# python sh04_inference_finetune.py --checkpoint="/users/PAS2062/delijingyic/project/wavegrad2/checkpoint/wavegrad2_10_01_02_klinR_999_epoch=33_loss=-3.663050651550293test.ckpt" --save_audio_dir=$WORK_DIR --resume_from=1059 --filename='klinR'

# python sh04_baseline_model_1059.py --checkpoint="/users/PAS2062/delijingyic/project/wavegrad2/checkpoint/wavegrad2_07_16_06_epoch=1755.ckpt" --save_audio_dir=$WORK_DIR --resume_from=1755

# python sh04_baseline_model_1059.py --checkpoint="/users/PAS2062/delijingyic/project/wavegrad2/checkpoint/wavegrad2_07_16_06_epoch=1756.ckpt" --save_audio_dir=$WORK_DIR --resume_from=1756
# srun --account=PAS1957 --ntasks=8 --gpus=1 --pty /bin/bash

# python sh04_inference_finetune.py --checkpoint="/users/PAS2062/delijingyic/project/wavegrad2/checkpoint/wavegrad2_10_27_00_epoch=3.ckpt" --save_audio_dir=$WORK_DIR --resume_from=3
