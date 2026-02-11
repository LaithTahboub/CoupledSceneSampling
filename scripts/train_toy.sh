#!/bin/bash
#SBATCH --job-name=css-toy-overfit
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=12:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_toy.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_toy.err

source /fs/nexus-scratch/ltahboub/CoupledSceneSampling/.venv/bin/activate

cd /fs/nexus-scratch/ltahboub/CoupledSceneSampling

python -m css.train \
    --scenes MegaScenes/Mysore_Palace \
    --output checkpoints/pose_sd_toy_overfit \
    --epochs 500 \
    --batch-size 2 \
    --lr 5e-5 \
    --max-pair-dist 2.0 \
    --max-triplets 50 \
    --save-every 50 \
    --cond-drop-prob 0.0 \
    --sample-cfg-scale 1.5 \
    --H 512 \
    --W 512 \
    --wandb-project CoupledSceneSampling \
    --wandb-name toy-overfit-v2 \
    --prompt "a photo of the Mysore palace"
