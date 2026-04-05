#!/bin/bash

#SBATCH --job-name=css-caption
#SBATCH --partition=vulcan-scavenger
#SBATCH --ntasks=1
#SBATCH --qos=vulcan-scavenger
#SBATCH --cpus-per-task=12
#SBATCH --mem=20gb
#SBATCH --time=2-0:00:00
#SBATCH --account=vulcan-jbhuang
#SBATCH --output=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/caption_%j.out
#SBATCH --error=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/caption_%j.err

# Batch-caption all MegaScenes images using a running vLLM server.
#
# Prerequisites: start the vLLM server first (scripts/vllm.sh), note the node.
#
# Usage:
#   VLLM_HOST=tron55 sbatch scripts/caption_megascenes.sh
#   VLLM_HOST=tron55 MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct sbatch scripts/caption_megascenes.sh

set -euo pipefail

ROOT="/vulcanscratch/ltahboub/CoupledSceneSampling"
VLLM_HOST=${VLLM_HOST:?Set VLLM_HOST to the node running vllm.sh (e.g. tron55)}
VLLM_PORT=${VLLM_PORT:-8000}
MODEL=${MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}

SCENES_FILE=${SCENES_FILE:-/fs/nexus-scratch/ltahboub/MegaScenes/scenes_colmap_ready.txt}
OUTPUT_DIR=${OUTPUT_DIR:-/fs/nexus-scratch/ltahboub/MegaScenesCaptions}

BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-12}
TEMPERATURE=${TEMPERATURE:-0.23}
MAX_TOKENS=${MAX_TOKENS:-100}

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi
cd "$ROOT"
mkdir -p logs

echo "=== MegaScenes Captioning ==="
echo "vLLM server: http://${VLLM_HOST}:${VLLM_PORT}"
echo "Model: ${MODEL}"
echo "Scenes file: ${SCENES_FILE}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

python -m css.data.caption_dataset \
    --scenes-file "$SCENES_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --api-base "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS"
