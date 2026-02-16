#!/bin/bash
# Backward-compatible wrapper.

#SBATCH --job-name=css-train-multiscene
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=4
#SBATCH --mem=64gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=4-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_multiscene.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_multiscene.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_multiscene.sh"
