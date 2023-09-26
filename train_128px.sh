#!/bin/sh
#
#
#SBATCH --account=nklab
#SBATCH --job-name=stylegan3  # The job name.
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --time=7-00:00
#SBATCH --exclude=ax[03-13]

ml load anaconda3-2019.03
ml gcc/10.4
ml cuda/11.6.2

cd /home/wg2361/xai-face-model/stylegan3

conda activate stylegan3

python -u train.py --gpus=8 --outdir=/scratch/nklab/projects/face_proj/models/stylegan3/ffhq_128 --cfg=stylegan3-r --data=/scratch/nklab/projects/face_proj/datasets/ffhq_128 --batch=32 --gamma=0.5 --cbase=16384