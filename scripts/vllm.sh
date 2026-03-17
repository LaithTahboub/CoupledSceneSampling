#!/bin/bash

#SBATCH --job-name=serve-vllm
#SBATCH --partition=vulcan-ampere
#SBATCH --qos=vulcan-high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --account=vulcan-jbhuang
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=1-0:00:00
#SBATCH --output=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/vllm_%j.out
#SBATCH --error=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/vllm_%j.err

set -euo pipefail

cd /vulcanscratch/ltahboub/CoupledSceneSampling
source /vulcanscratch/ltahboub/CoupledSceneSampling/.venv/bin/activate

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 4096 \
    --limit-mm-per-prompt '{"image": 1}'