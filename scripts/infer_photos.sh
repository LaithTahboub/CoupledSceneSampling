#!/bin/bash

#SBATCH --job-name=css-infer-photos
#SBATCH --partition=tron
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=1-0:00:00
#SBATCH --output=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/infer_photos_%j.out
#SBATCH --error=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/infer_photos_%j.err

set -euo pipefail

source /vulcanscratch/ltahboub/CoupledSceneSampling/.venv/bin/activate

uv run -m css.inference.infer_photos \
        --ref1 /vulcanscratch/ltahboub/CoupledSceneSampling/misc/cathedral1.jpeg --ref2 /vulcanscratch/ltahboub/CoupledSceneSampling/misc/cathedral2.jpeg --target /vulcanscratch/ltahboub/CoupledSceneSampling/misc/cathedral3.jpeg \
        --checkpoint checkpoints/pose_sd_v4/unet_latest.pt \
        --prompt ""