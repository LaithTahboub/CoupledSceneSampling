#!/bin/bash
# Single-scene training with a held-out image split.

#SBATCH --job-name=css-train-single
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=3-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_single_scene.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_single_scene.err

set -euo pipefail

ROOT=${ROOT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling}
SCENE=${SCENE:-MegaScenes/Mysore_Palace}

TEST_RATIO=${TEST_RATIO:-0.10}
TRAIN_RATIO=${TRAIN_RATIO:-1.0}
SEED=${SEED:-42}

SCENE_BASENAME=$(basename "$SCENE")
SCENE_TAG=$(echo "$SCENE_BASENAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g')
SCENE_TEXT=$(echo "$SCENE_BASENAME" | tr '_' ' ')

SPLIT_TAG="test${TEST_RATIO}_train${TRAIN_RATIO}_seed${SEED}"
SPLIT_TAG=${SPLIT_TAG//./p}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/${SCENE_TAG}_${SPLIT_TAG}}

EPOCHS=${EPOCHS:-100}
OUTPUT=${OUTPUT:-checkpoints/pose_sd_${SCENE_TAG}_${SPLIT_TAG}}
RUN_NAME=${RUN_NAME:-${SCENE_TAG}-${SPLIT_TAG}}
WANDB_ID=${WANDB_ID:-}

PROMPT=${PROMPT:-"a photo of ${SCENE_TEXT}"}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.0}
MIN_DIR_SIM=${MIN_DIR_SIM:-0.3}
MIN_REF_SPACING=${MIN_REF_SPACING:-0.3}
MAX_TRIPLETS=${MAX_TRIPLETS:-1000000}

BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-0}
LR=${LR:-1e-5}
WARMUP_STEPS=${WARMUP_STEPS:-500}
EMA_DECAY=${EMA_DECAY:-0.9999}
SAVE_EVERY=${SAVE_EVERY:-4}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-5}
COND_DROP_PROB=${COND_DROP_PROB:-0.1}
SAMPLE_CFG_SCALE=${SAMPLE_CFG_SCALE:-7.5}
MIN_TIMESTEP=${MIN_TIMESTEP:-0}
MAX_TIMESTEP=${MAX_TIMESTEP:-}

H=${H:-512}
W=${W:-512}

source "$ROOT/.venv/bin/activate"
cd "$ROOT"

python -m css.make_scene_split \
    --scene "$SCENE" \
    --output-dir "$SPLIT_DIR" \
    --test-ratio "$TEST_RATIO" \
    --train-ratio "$TRAIN_RATIO" \
    --seed "$SEED"

TRAIN_ARGS=(
    --scenes "$SCENE"
    --output "$OUTPUT"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --warmup-steps "$WARMUP_STEPS"
    --ema-decay "$EMA_DECAY"
    --max-pair-dist "$MAX_PAIR_DIST"
    --min-dir-sim "$MIN_DIR_SIM"
    --min-ref-spacing "$MIN_REF_SPACING"
    --max-triplets "$MAX_TRIPLETS"
    --save-every "$SAVE_EVERY"
    --keep-checkpoints "$KEEP_CHECKPOINTS"
    --prompt "$PROMPT"
    --unet-train-mode cond
    --cond-drop-prob "$COND_DROP_PROB"
    --sample-cfg-scale "$SAMPLE_CFG_SCALE"
    --min-timestep "$MIN_TIMESTEP"
    --exclude-image-list "$SPLIT_DIR/test_images.txt"
    --target-include-image-list "$SPLIT_DIR/train_images.txt"
    --reference-include-image-list "$SPLIT_DIR/train_images.txt"
    --H "$H"
    --W "$W"
    --wandb-project CoupledSceneSampling
    --wandb-name "$RUN_NAME"
)

if [[ -n "$MAX_TIMESTEP" ]]; then
    TRAIN_ARGS+=(--max-timestep "$MAX_TIMESTEP")
fi
if [[ -n "$WANDB_ID" ]]; then
    TRAIN_ARGS+=(--wandb-id "$WANDB_ID")
fi

python -m css.train "${TRAIN_ARGS[@]}"
