#!/bin/bash
# Train on a large subset while holding out test images that are never seen in training.

#SBATCH --job-name=css-train-mysore-split
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=3-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_split.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_split.err

set -euo pipefail

ROOT=/fs/nexus-scratch/ltahboub/CoupledSceneSampling
SCENE=MegaScenes/Mysore_Palace
SPLIT_DIR=$ROOT/splits/mysore_palace_test10_seed42

source $ROOT/.venv/bin/activate
cd $ROOT

python -m css.make_scene_split \
    --scene $SCENE \
    --output-dir $SPLIT_DIR \
    --test-ratio 0.10 \
    --train-ratio 1.0 \
    --seed 42

python -m css.train \
    --scenes $SCENE \
    --output checkpoints/pose_sd_mysore_split_v1 \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-5 \
    --max-pair-dist 2.0 \
    --max-triplets 1000000 \
    --save-every 20 \
    --prompt "a photo of the Mysore palace" \
    --unet-train-mode cond \
    --cond-drop-prob 0.1 \
    --sample-cfg-scale 1.0 \
    --min-timestep 0 \
    --exclude-image-list $SPLIT_DIR/test_images.txt \
    --target-include-image-list $SPLIT_DIR/train_images.txt \
    --reference-include-image-list $SPLIT_DIR/train_images.txt \
    --H 512 \
    --W 512 \
    --wandb-project CoupledSceneSampling \
    --wandb-name mysore-palace-split-v1
