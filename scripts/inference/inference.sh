#!/bin/bash
#SBATCH --job-name=inference-stable-diffusion
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=1
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=24:00:00

source /fs/nexus-scratch/ltahboub/CoupledSceneSampling/.venv/bin/activate
uv run --active /fs/nexus-scratch/ltahboub/CoupledSceneSampling/scripts/inference/SD1.5.py