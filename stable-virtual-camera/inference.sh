#!/bin/bash
#SBATCH --job-name=inference-coupled-diffusion
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=1
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=24:00:00

source /fs/nexus-scratch/ltahboub/CoupledSceneSampling/.venv/bin/activate
uv run --active coupled_diffusion.py --inputs /fs/nexus-scratch/ltahboub/CoupledSceneSampling/MegaScenes/Mysore_Palace/images/commons/East_side_of_the_Mysore_Palace/0/pictures/Mysore_Palace_2006.jpg:0 --trajectory orbit --frames 25 --edit-strength 3