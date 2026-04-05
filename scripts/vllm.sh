#!/bin/bash

#SBATCH --job-name=serve-vllm
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90gb
#SBATCH --account=vulcan-jbhuang
#SBATCH --gres=gpu:h200-sxm:2
#SBATCH --time=2-0:00:00
#SBATCH --output=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/vllm_%j.out
#SBATCH --error=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/vllm_%j.err

set -euo pipefail

cd /vulcanscratch/ltahboub/CoupledSceneSampling

export HF_HOME=/vulcanscratch/ltahboub/.hf_cache
export APPTAINER_CACHEDIR=/vulcanscratch/ltahboub/.apptainer_cache
export APPTAINER_TMPDIR=/vulcanscratch/ltahboub/.apptainer_tmp
export OMP_NUM_THREADS=1

apptainer exec --nv \
    --bind /vulcanscratch/ltahboub:/vulcanscratch/ltahboub \
    vllm-openai_latest.sif \
    python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --mm-encoder-tp-mode data \
    --enable-expert-parallel \
    --async-scheduling \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95 \
    --limit-mm-per-prompt.image 1 \
    --limit-mm-per-prompt.video 0 \
    --mm-processor-cache-gb 0 \
    --trust-remote-code