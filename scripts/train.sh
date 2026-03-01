#!/bin/bash
# Default training entrypoint: multiscene training with a held-out scene split.

#SBATCH --job-name=css-train
#SBATCH --partition=vulcan-ampere#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=3-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train2.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train2.err

set -euo pipefail

ROOT=${ROOT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling}
MEGASCENES_ROOT=${MEGASCENES_ROOT:-$ROOT/MegaScenes}
SCENES_FILE=${SCENES_FILE:-$MEGASCENES_ROOT/scenes_colmap_ready.txt}

TEST_RATIO=${TEST_RATIO:-0.10}
SEED=${SEED:-315}
MAX_SCENES=${MAX_SCENES:-201}
MIN_TRAIN_SCENES=${MIN_TRAIN_SCENES:-200}
MIN_READY_SCENES=${MIN_READY_SCENES:-201}

SPLIT_TAG="test${TEST_RATIO}_seed${SEED}"
SPLIT_TAG=${SPLIT_TAG//./p}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/multiscene_scenes_${SPLIT_TAG}}

EPOCHS=${EPOCHS:-100}
OUTPUT=${OUTPUT:-checkpoints/pose_sd_multiscene_${SPLIT_TAG}}
RUN_NAME=${RUN_NAME:-multiscene-${SPLIT_TAG}-bigger}
WANDB_ID=${WANDB_ID:-}

PROMPT=${PROMPT:-""}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-""}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.5}
MIN_PAIR_IOU=${MIN_PAIR_IOU:-0.15}
MIN_REF_SPACING=${MIN_REF_SPACING:-0.25}
MAX_TRIPLETS=${MAX_TRIPLETS:-500}

BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
LR=${LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.999}
ADAM_EPS=${ADAM_EPS:-1e-8}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
EMA_DECAY=${EMA_DECAY:-0.9999}
GRAD_CLIP=${GRAD_CLIP:-1.0}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
XFORMERS_ATTENTION=${XFORMERS_ATTENTION:-0}
SAVE_EVERY=${SAVE_EVERY:-5}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-3}
COND_DROP_PROB=${COND_DROP_PROB:-0.15}
SAMPLE_CFG_SCALE=${SAMPLE_CFG_SCALE:-3.5}
SAMPLE_STEPS=${SAMPLE_STEPS:-50}
SAMPLE_EVERY=${SAMPLE_EVERY:-1}
NOISE_OFFSET=${NOISE_OFFSET:-0.05}
MIN_SNR_GAMMA=${MIN_SNR_GAMMA:-5.0}
MIN_TIMESTEP=${MIN_TIMESTEP:-20}
MAX_TIMESTEP=${MAX_TIMESTEP:-980}

H=${H:-512}
W=${W:-512}

source "$ROOT/.venv/bin/activate"
cd "$ROOT"

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

TRAIN_ARGS=(
    --scenes-file "$TRAIN_SCENES_FILE"
    --output "$OUTPUT"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --prefetch-factor "$PREFETCH_FACTOR"
    --grad-accum-steps "$GRAD_ACCUM_STEPS"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --adam-beta1 "$ADAM_BETA1"
    --adam-beta2 "$ADAM_BETA2"
    --adam-eps "$ADAM_EPS"
    --warmup-steps "$WARMUP_STEPS"
    --ema-decay "$EMA_DECAY"
    --grad-clip "$GRAD_CLIP"
    --mixed-precision "$MIXED_PRECISION"
    --seed "$SEED"
    --max-pair-dist "$MAX_PAIR_DIST"
    --min-pair-iou "$MIN_PAIR_IOU"
    --min-ref-spacing "$MIN_REF_SPACING"
    --max-triplets "$MAX_TRIPLETS"
    --save-every "$SAVE_EVERY"
    --keep-checkpoints "$KEEP_CHECKPOINTS"
    --prompt "$PROMPT"
    --unet-train-mode cond
    --cond-drop-prob "$COND_DROP_PROB"
    --noise-offset "$NOISE_OFFSET"
    --min-snr-gamma "$MIN_SNR_GAMMA"
    --sample-cfg-scale "$SAMPLE_CFG_SCALE"
    --sample-steps "$SAMPLE_STEPS"
    --sample-every "$SAMPLE_EVERY"
    --min-timestep "$MIN_TIMESTEP"
    --max-timestep "$MAX_TIMESTEP"
    --H "$H"
    --W "$W"
    --wandb-project CoupledSceneSampling
    --wandb-name "$RUN_NAME"
)

if [[ -n "$PROMPT_TEMPLATE" ]]; then
    TRAIN_ARGS+=(--prompt-template "$PROMPT_TEMPLATE")
fi
if [[ -f "$TEST_SCENES_FILE" ]]; then
    TRAIN_ARGS+=(--test-scenes-file "$TEST_SCENES_FILE")
fi
if [[ -n "$WANDB_ID" ]]; then
    TRAIN_ARGS+=(--wandb-id "$WANDB_ID")
fi
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
    TRAIN_ARGS+=(--gradient-checkpointing)
fi
if [[ "$XFORMERS_ATTENTION" == "1" ]]; then
    TRAIN_ARGS+=(--xformers-attention)
fi

python -m css.train "${TRAIN_ARGS[@]}"
