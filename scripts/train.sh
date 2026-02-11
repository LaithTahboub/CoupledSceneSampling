#!/bin/bash
#SBATCH --job-name=css-train-pose-sd-new
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=3-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train.err

source /fs/nexus-scratch/ltahboub/CoupledSceneSampling/.venv/bin/activate

cd /fs/nexus-scratch/ltahboub/CoupledSceneSampling

python -m css.train \
    --scenes MegaScenes/Mysore_Palace \
    --output checkpoints/pose_sd_mysore_v4 \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-5 \
    --max-pair-dist 2.0 \
    --max-triplets 10000 \
    --save-every 5 \
    --H 512 \
    --W 512 \
    --wandb-project CoupledSceneSampling \
    --wandb-name mysore-palace-v4
