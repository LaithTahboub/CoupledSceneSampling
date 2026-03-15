#!/bin/bash
# PoseSD training: 3-view (ref1, ref2, target) with Plucker ray conditioning.
# Multi-GPU via torchrun (8x A6000).

#SBATCH --job-name=css-pose-sd
#SBATCH --partition=vulcan-scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100gb
#SBATCH --gres=gpu:rtxa6000:2
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-scavenger
#SBATCH --time=3-0:00:00
#SBATCH --output=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/train_pose_sd_%j.out
#SBATCH --error=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/train_pose_sd_%j.err

set -euo pipefail

ROOT="/vulcanscratch/ltahboub/CoupledSceneSampling"
SCENES_FILE=${SCENES_FILE:-"$ROOT/MegaScenes/scenes_colmap_ready.txt"}
SCENES=${SCENES:-}

OUTPUT=${OUTPUT:-$ROOT/checkpoints/pose_sd_v6}
SEED=${SEED:-31}

# --- Training ---
TOTAL_STEPS=${TOTAL_STEPS:-200000}
PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
# Effective batch size: 2 * 8 GPUs * 8 accum = 128
LR=${LR:-1e-4}
TRAIN_MODE=${TRAIN_MODE:-cond}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}

# --- Data ---
H=${H:-512}
W=${W:-512}
MAX_TRIPLETS_PER_SCENE=${MAX_TRIPLETS_PER_SCENE:-111}
MIN_POINTS_PER_IMAGE=${MIN_POINTS_PER_IMAGE:-400}
MIN_ORIENTATION_DOT=${MIN_ORIENTATION_DOT:-0.5}
MAX_FOCAL_LENGTH_RATIO=${MAX_FOCAL_LENGTH_RATIO:-2.0}
MIN_REF_COVISIBILITY=${MIN_REF_COVISIBILITY:-0.10}
MAX_REF_COVISIBILITY=${MAX_REF_COVISIBILITY:-0.70}
NEAR_DUPLICATE_THRESHOLD=${NEAR_DUPLICATE_THRESHOLD:-0.82}

# --- Conditioning dropout ---
COND_BOTH_KEPT=${COND_BOTH_KEPT:-0.85}
COND_ONE_DROPPED=${COND_ONE_DROPPED:-0.10}
COND_BOTH_DROPPED=${COND_BOTH_DROPPED:-0.05}

# --- Bucket ratios ---
EASY_RATIO=${EASY_RATIO:-0.50}
MEDIUM_RATIO=${MEDIUM_RATIO:-0.35}
HARD_RATIO=${HARD_RATIO:-0.15}

# --- Split ---
TEST_SCENES_PCT=${TEST_SCENES_PCT:-5.0}
TEST_TARGETS_PER_SCENE=${TEST_TARGETS_PER_SCENE:-1}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/pose_sd_seed${SEED}}

# --- Checkpoints & validation ---
SAVE_EVERY=${SAVE_EVERY:-8000}
VAL_EVERY=${VAL_EVERY:-3000}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-5}
VAL_SAMPLE_STEPS=${VAL_SAMPLE_STEPS:-50}
VAL_CFG_SCALE=${VAL_CFG_SCALE:-3.0}

# --- EMA ---
EMA_DECAY=${EMA_DECAY:-0.9999}

# --- Multi-GPU ---
NUM_GPUS=${NUM_GPUS:-2}
NUM_WORKERS=${NUM_WORKERS:-4}

# Resume
RESUME=${RESUME:-}

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi
cd "$ROOT"
mkdir -p logs

ARGS=(
    --output "$OUTPUT"
    --split-dir "$SPLIT_DIR"
    --seed "$SEED"
    --total-steps "$TOTAL_STEPS"
    --per-gpu-batch-size "$PER_GPU_BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --lr "$LR"
    --weight-decay 0.01
    --grad-clip 1.0
    --warmup-steps "$WARMUP_STEPS"
    --lr-scheduler "$LR_SCHEDULER"
    --train-mode "$TRAIN_MODE"
    --gradient-checkpointing
    --cond-both-kept "$COND_BOTH_KEPT"
    --cond-one-dropped "$COND_ONE_DROPPED"
    --cond-both-dropped "$COND_BOTH_DROPPED"
    --easy-ratio "$EASY_RATIO"
    --medium-ratio "$MEDIUM_RATIO"
    --hard-ratio "$HARD_RATIO"
    --max-triplets-per-scene "$MAX_TRIPLETS_PER_SCENE"
    --min-points-per-image "$MIN_POINTS_PER_IMAGE"
    --min-orientation-dot "$MIN_ORIENTATION_DOT"
    --max-focal-length-ratio "$MAX_FOCAL_LENGTH_RATIO"
    --min-ref-covisibility "$MIN_REF_COVISIBILITY"
    --max-ref-covisibility "$MAX_REF_COVISIBILITY"
    --near-duplicate-threshold "$NEAR_DUPLICATE_THRESHOLD"
    --test-scenes-pct "$TEST_SCENES_PCT"
    --test-targets-per-scene "$TEST_TARGETS_PER_SCENE"
    --save-every-steps "$SAVE_EVERY"
    --val-every-steps "$VAL_EVERY"
    --keep-checkpoints "$KEEP_CHECKPOINTS"
    --val-sample-steps "$VAL_SAMPLE_STEPS"
    --val-cfg-scale "$VAL_CFG_SCALE"
    --ema-decay "$EMA_DECAY"
    --H "$H"
    --W "$W"
    --num-workers "$NUM_WORKERS"
    --mixed-precision bf16
)

if [[ -n "$SCENES" ]]; then
    # shellcheck disable=SC2206
    SCENE_LIST=($SCENES)
    ARGS+=(--scenes "${SCENE_LIST[@]}")
elif [[ -f "$SCENES_FILE" ]]; then
    ARGS+=(--scenes-file "$SCENES_FILE")
else
    echo "Provide SCENES (space-separated) or a valid SCENES_FILE."
    echo "SCENES_FILE currently points to: $SCENES_FILE"
    exit 1
fi

if [[ -n "$RESUME" ]]; then
    ARGS+=(--resume-from "$RESUME")
fi

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="${MASTER_PORT:-29501}" \
    -m css.train.train_pose_sd "${ARGS[@]}"
