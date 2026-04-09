#!/bin/bash
# RelightFlux LoRA training: 3-view (ref1, ref2, target) with Plucker ray conditioning.
# Much faster and lighter than full fine-tuning (~200MB trainable vs ~24GB).
# Multi-GPU via torchrun on Flux.1-dev backbone (~12B params, LoRA adapters only).

#SBATCH --job-name=css-relight-flux-lora
#SBATCH --partition=vulcan-scavenger
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=256gb
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-scavenger
#SBATCH --time=3-00:00:00
#SBATCH --output=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/train_relight_flux_lora_%j.out
#SBATCH --error=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/train_relight_flux_lora_%j.err

set -euo pipefail

ROOT="/vulcanscratch/ltahboub/CoupledSceneSampling"
SCENES_FILE=${SCENES_FILE:-"/fs/nexus-scratch/ltahboub/MegaScenes/scenes_colmap_ready.txt"}
SCENES=${SCENES:-}

RUN_NAME=${RUN_NAME:-relight_flux_lora_v1}
OUTPUT=${OUTPUT:-$ROOT/checkpoints/${RUN_NAME}}
SEED=${SEED:-7}

# - Training -
# LoRA is much lighter — can use larger batch and fewer accum steps
TOTAL_STEPS=${TOTAL_STEPS:-30000}
PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-4}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-0}

LR=${LR:-1e-4}
TRAIN_MODE=${TRAIN_MODE:-lora}
WARMUP_STEPS=${WARMUP_STEPS:-500}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}

# - LoRA config -
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}

# - Data -
# 512x512 for training (same as full fine-tune)
H=${H:-512}
W=${W:-512}
MAX_TRIPLETS_PER_SCENE=${MAX_TRIPLETS_PER_SCENE:-300000}
MIN_POINTS_PER_IMAGE=${MIN_POINTS_PER_IMAGE:-300}
MIN_ORIENTATION_DOT=${MIN_ORIENTATION_DOT:-0.4}
MAX_FOCAL_LENGTH_RATIO=${MAX_FOCAL_LENGTH_RATIO:-2.0}
MIN_REF_COVISIBILITY=${MIN_REF_COVISIBILITY:-0.15}
MAX_REF_COVISIBILITY=${MAX_REF_COVISIBILITY:-0.65}
NEAR_DUPLICATE_THRESHOLD=${NEAR_DUPLICATE_THRESHOLD:-0.82}
MIN_TARGETS_PER_SCENE=${MIN_TARGETS_PER_SCENE:-10}

# - Conditioning dropout -
COND_BOTH_KEPT=${COND_BOTH_KEPT:-0.85}
COND_ONE_DROPPED=${COND_ONE_DROPPED:-0.10}
COND_BOTH_DROPPED=${COND_BOTH_DROPPED:-0.05}

# - Captioning -
CAPTION_DIR=${CAPTION_DIR:-/fs/nexus-scratch/ltahboub/MegaScenesCaptions}
TEXT_DROP_PROB=${TEXT_DROP_PROB:-0.1}

# - Data augmentation -
IDENTITY_AUG_PROB=${IDENTITY_AUG_PROB:-0.03}
RANDOM_CROP_PROB=${RANDOM_CROP_PROB:-0.15}

# - Bucket ratios -
EASY_RATIO=${EASY_RATIO:-0.20}
MEDIUM_RATIO=${MEDIUM_RATIO:-0.60}
HARD_RATIO=${HARD_RATIO:-0.20}

# - Split -
TEST_SCENES_PCT=${TEST_SCENES_PCT:-50.0}
TEST_TARGETS_PER_SCENE=${TEST_TARGETS_PER_SCENE:-0}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/${RUN_NAME}_seed${SEED}}

# - Checkpoints & validation -
# LoRA trains faster, but step-3 previews are misleading for this model:
# they often show denoiser-path artifacts before the multi-view conditioning
# has adapted at all. Start validation later so the default run surfaces
# meaningful images instead of near-initialization samples.
SAVE_EVERY=${SAVE_EVERY:-250}
VAL_EVERY=${VAL_EVERY:-250}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-2}
VAL_SAMPLE_STEPS=${VAL_SAMPLE_STEPS:-28}
VAL_CFG_SCALE=${VAL_CFG_SCALE:-3.0}
VAL_CFG_TEXT=${VAL_CFG_TEXT:-5.0}
VAL_SEEDS_PER_SAMPLE=${VAL_SEEDS_PER_SAMPLE:-2}

# - EMA -
# LoRA adapts quickly; a very sticky EMA makes early previews/checkpoints look
# almost identical to initialization. Use a faster shadow by default.
EMA_DECAY=${EMA_DECAY:-0.995}
USE_EMA=${USE_EMA:-1}

# - W&B -
WANDB_MODE=${WANDB_MODE:-online}
WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}

# - Multi-GPU -
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
    --lora-rank "$LORA_RANK"
    --lora-alpha "$LORA_ALPHA"
    --lora-dropout "$LORA_DROPOUT"
    --cond-both-kept "$COND_BOTH_KEPT"
    --cond-one-dropped "$COND_ONE_DROPPED"
    --cond-both-dropped "$COND_BOTH_DROPPED"
    --text-drop-prob "$TEXT_DROP_PROB"
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
    --min-targets-per-scene "$MIN_TARGETS_PER_SCENE"
    --identity-aug-prob "$IDENTITY_AUG_PROB"
    --random-crop-prob "$RANDOM_CROP_PROB"
    --test-scenes-pct "$TEST_SCENES_PCT"
    --test-targets-per-scene "$TEST_TARGETS_PER_SCENE"
    --save-every-steps "$SAVE_EVERY"
    --val-every-steps "$VAL_EVERY"
    --keep-checkpoints "$KEEP_CHECKPOINTS"
    --val-sample-steps "$VAL_SAMPLE_STEPS"
    --val-cfg-scale "$VAL_CFG_SCALE"
    --val-cfg-text "$VAL_CFG_TEXT"
    --val-seeds-per-sample "$VAL_SEEDS_PER_SAMPLE"
    --ema-decay "$EMA_DECAY"
    --H "$H"
    --W "$W"
    --num-workers "$NUM_WORKERS"
    --mixed-precision bf16
    --wandb-mode "$WANDB_MODE"
    --wandb-init-timeout "$WANDB_INIT_TIMEOUT"
)

if [[ "$USE_EMA" == "0" ]]; then
    ARGS+=(--no-ema)
fi

if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
    ARGS+=(--gradient-checkpointing)
fi

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

if [[ -n "$CAPTION_DIR" ]]; then
    ARGS+=(--caption-dir "$CAPTION_DIR")
fi

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="${MASTER_PORT:-29504}" \
    -m css.train.train_relight_flux "${ARGS[@]}"
