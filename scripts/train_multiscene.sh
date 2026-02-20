#!/bin/bash
# Train on many COLMAP-ready scenes with per-scene prompt template.

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

ROOT=${ROOT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling}
MEGASCENES_ROOT=${MEGASCENES_ROOT:-$ROOT/MegaScenes}
SCENES_FILE=${SCENES_FILE:-$MEGASCENES_ROOT/scenes_colmap_ready.txt}
MIN_SCENES=${MIN_SCENES:-100}

EPOCHS=${EPOCHS:-100}
OUTPUT=${OUTPUT:-checkpoints/pose_sd_multiscene_v1}
RUN_NAME=${RUN_NAME:-multiscene-v1}

PROMPT=${PROMPT:-"a photo of a scene"}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"a photo of {scene}"}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.5}
MIN_DIR_SIM=${MIN_DIR_SIM:-0.2}
MIN_REF_SPACING=${MIN_REF_SPACING:-0.25}
MAX_TRIPLETS=${MAX_TRIPLETS:-3000}

BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-5e-6}
SAVE_EVERY=${SAVE_EVERY:-4}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-5}
COND_DROP_PROB=${COND_DROP_PROB:-0.1}
SAMPLE_CFG_SCALE=${SAMPLE_CFG_SCALE:-7.5}

H=${H:-512}
W=${W:-512}

source $ROOT/.venv/bin/activate
cd $ROOT

if [[ ! -f "$SCENES_FILE" ]]; then
    echo "Scenes file not found: $SCENES_FILE"
    exit 1
fi

READY_COUNT=$(wc -l < "$SCENES_FILE" || echo 0)
echo "COLMAP-ready scenes: $READY_COUNT"
if (( READY_COUNT < MIN_SCENES )); then
    echo "Need at least $MIN_SCENES scenes before training."
    exit 1
fi

python -m css.train \
    --scenes-file "$SCENES_FILE" \
    --output "$OUTPUT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --lr "$LR" \
    --max-pair-dist "$MAX_PAIR_DIST" \
    --min-dir-sim "$MIN_DIR_SIM" \
    --min-ref-spacing "$MIN_REF_SPACING" \
    --max-triplets "$MAX_TRIPLETS" \
    --save-every "$SAVE_EVERY" \
    --keep-checkpoints "$KEEP_CHECKPOINTS" \
    --prompt "$PROMPT" \
    --prompt-template "$PROMPT_TEMPLATE" \
    --unet-train-mode cond \
    --cond-drop-prob "$COND_DROP_PROB" \
    --sample-cfg-scale "$SAMPLE_CFG_SCALE" \
    --min-timestep 0 \
    --H "$H" \
    --W "$W" \
    --wandb-project CoupledSceneSampling \
    --wandb-name "$RUN_NAME"
