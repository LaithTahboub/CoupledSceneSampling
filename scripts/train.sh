#!/bin/bash
# Default training entrypoint: multiscene training with a held-out scene split.

#SBATCH --job-name=css-train
#SBATCH --partition=tron
#SBATCH --ntasks=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=3-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train.err

set -euo pipefail

ROOT=${ROOT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling}
MEGASCENES_ROOT=${MEGASCENES_ROOT:-$ROOT/MegaScenes}
SCENES_FILE=${SCENES_FILE:-$MEGASCENES_ROOT/scenes_colmap_ready.txt}
TRIPLETS_MANIFEST=${TRIPLETS_MANIFEST:-}

TEST_RATIO=${TEST_RATIO:-0.10}
SEED=${SEED:-42}
MAX_SCENES=${MAX_SCENES:-100}
MIN_TRAIN_SCENES=${MIN_TRAIN_SCENES:-50}
MIN_READY_SCENES=${MIN_READY_SCENES:-100}

SPLIT_TAG="test${TEST_RATIO}_seed${SEED}"
SPLIT_TAG=${SPLIT_TAG//./p}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/multiscene_scenes_${SPLIT_TAG}}

EPOCHS=${EPOCHS:-100}
OUTPUT=${OUTPUT:-checkpoints/pose_sd_multiscene_${SPLIT_TAG}}
RUN_NAME=${RUN_NAME:-multiscene-${SPLIT_TAG}}
WANDB_ID=${WANDB_ID:-}

PROMPT=${PROMPT:-"a photo of a scene"}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"a photo of {scene}"}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.5}
MIN_DIR_SIM=${MIN_DIR_SIM:-0.2}
MIN_REF_SPACING=${MIN_REF_SPACING:-0.25}
MAX_TRIPLETS=${MAX_TRIPLETS:-8}

BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-5e-6}
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

TRAIN_INPUT_ARGS=()
if [[ -n "$TRIPLETS_MANIFEST" ]]; then
    if [[ ! -f "$TRIPLETS_MANIFEST" ]]; then
        echo "Triplets manifest not found: $TRIPLETS_MANIFEST"
        exit 1
    fi
    echo "Training from triplets manifest: $TRIPLETS_MANIFEST"
    TRAIN_INPUT_ARGS=(--triplets-manifest "$TRIPLETS_MANIFEST")
else
    if [[ ! -f "$SCENES_FILE" ]]; then
        echo "Scenes file not found: $SCENES_FILE"
        exit 1
    fi

    READY_COUNT=$(wc -l < "$SCENES_FILE" || echo 0)
    echo "Ready scenes in list: $READY_COUNT"
    if (( READY_COUNT < MIN_READY_SCENES )); then
        echo "Need at least $MIN_READY_SCENES scenes in SCENES_FILE before training."
        exit 1
    fi

    SPLIT_ARGS=(
        --scenes-file "$SCENES_FILE"
        --output-dir "$SPLIT_DIR"
        --test-ratio "$TEST_RATIO"
        --seed "$SEED"
        --min-train-scenes "$MIN_TRAIN_SCENES"
    )
    if [[ -n "$MAX_SCENES" ]]; then
        SPLIT_ARGS+=(--max-scenes "$MAX_SCENES")
    fi

    python -m css.make_scenes_split "${SPLIT_ARGS[@]}"

    TRAIN_SCENES_FILE="$SPLIT_DIR/train_scenes.txt"
    TEST_SCENES_FILE="$SPLIT_DIR/test_scenes.txt"
    if [[ ! -f "$TRAIN_SCENES_FILE" || ! -f "$TEST_SCENES_FILE" ]]; then
        echo "Split creation failed in $SPLIT_DIR"
        exit 1
    fi

    TRAIN_SCENES_COUNT=$(wc -l < "$TRAIN_SCENES_FILE" || echo 0)
    TEST_SCENES_COUNT=$(wc -l < "$TEST_SCENES_FILE" || echo 0)
    echo "Train scenes: $TRAIN_SCENES_COUNT"
    echo "Test scenes:  $TEST_SCENES_COUNT"
    echo "Split dir: $SPLIT_DIR"
    if (( TRAIN_SCENES_COUNT < 1 )); then
        echo "Split has no train scenes: $TRAIN_SCENES_FILE"
        exit 1
    fi

    TRAIN_INPUT_ARGS=(--scenes-file "$TRAIN_SCENES_FILE")
fi

TRAIN_ARGS=(
    "${TRAIN_INPUT_ARGS[@]}"
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
    --H "$H"
    --W "$W"
    --wandb-project CoupledSceneSampling
    --wandb-name "$RUN_NAME"
)

if [[ -n "$PROMPT_TEMPLATE" ]]; then
    TRAIN_ARGS+=(--prompt-template "$PROMPT_TEMPLATE")
fi
if [[ -n "$MAX_TIMESTEP" ]]; then
    TRAIN_ARGS+=(--max-timestep "$MAX_TIMESTEP")
fi
if [[ -n "$WANDB_ID" ]]; then
    TRAIN_ARGS+=(--wandb-id "$WANDB_ID")
fi

python -m css.train "${TRAIN_ARGS[@]}"
