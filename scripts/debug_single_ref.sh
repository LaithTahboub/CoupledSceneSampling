#!/bin/bash
# Single-reference (no Plucker) debug training with CAT3D-style cross-view attention.

#SBATCH --job-name=css-debug-single-ref
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=3-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/debug_single_ref.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/debug_single_ref.err

set -euo pipefail

ROOT=${ROOT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling}
MEGASCENES_ROOT=${MEGASCENES_ROOT:-$ROOT/MegaScenes}
SCENES_FILE=${SCENES_FILE:-$MEGASCENES_ROOT/scenes_colmap_ready.txt}
SCENES=${SCENES:-}
SCENE=${SCENE:-MegaScenes/Mysore_Palace}

SEED=${SEED:-315}
OUTPUT=${OUTPUT:-checkpoints/single_ref_debug_seed${SEED}}

EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
LR=${LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
GRAD_CLIP=${GRAD_CLIP:-1.0}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
XFORMERS_ATTENTION=${XFORMERS_ATTENTION:-0}
TRAIN_MODE=${TRAIN_MODE:-cond}

COND_DROP_PROB=${COND_DROP_PROB:-0.15}
NOISE_OFFSET=${NOISE_OFFSET:-0.05}
MIN_SNR_GAMMA=${MIN_SNR_GAMMA:-5.0}
MIN_TIMESTEP=${MIN_TIMESTEP:-20}
MAX_TIMESTEP=${MAX_TIMESTEP:-980}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-a photo of {scene}}

# Debug pair mining constraints: keep moderate overlap to avoid copy behavior.
MIN_PAIR_IOU=${MIN_PAIR_IOU:-0.18}
MAX_PAIR_IOU=${MAX_PAIR_IOU:-0.62}
MIN_PAIR_DISTANCE=${MIN_PAIR_DISTANCE:-0.20}
MAX_PAIR_DISTANCE=${MAX_PAIR_DISTANCE:-2.2}
MIN_VIEW_COS=${MIN_VIEW_COS:-0.90}
MIN_ROTATION_DEG=${MIN_ROTATION_DEG:-3.0}
MAX_ROTATION_DEG=${MAX_ROTATION_DEG:-35.0}
MAX_FOCAL_RATIO=${MAX_FOCAL_RATIO:-1.35}
PREFILTER_TOPK=${PREFILTER_TOPK:-48}
TARGETS_PER_REF=${TARGETS_PER_REF:-2}
MAX_PAIRS_PER_SCENE=${MAX_PAIRS_PER_SCENE:-128}
VAL_RATIO=${VAL_RATIO:-0.10}

SAMPLE_EVERY_EPOCHS=${SAMPLE_EVERY_EPOCHS:-1}
SAMPLE_STEPS=${SAMPLE_STEPS:-50}
SAMPLE_CFG_SCALE=${SAMPLE_CFG_SCALE:-3.5}
NUM_DEBUG_PAIRS=${NUM_DEBUG_PAIRS:-4}
NUM_DEBUG_SAMPLES=${NUM_DEBUG_SAMPLES:-6}
SAMPLE_SEED_BASE=${SAMPLE_SEED_BASE:-1234}

H=${H:-512}
W=${W:-512}

source "$ROOT/.venv/bin/activate"
cd "$ROOT"

ARGS=(
    --output "$OUTPUT"
    --seed "$SEED"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --prefetch-factor "$PREFETCH_FACTOR"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --warmup-steps "$WARMUP_STEPS"
    --grad-clip "$GRAD_CLIP"
    --mixed-precision "$MIXED_PRECISION"
    --train-mode "$TRAIN_MODE"
    --cond-drop-prob "$COND_DROP_PROB"
    --noise-offset "$NOISE_OFFSET"
    --min-snr-gamma "$MIN_SNR_GAMMA"
    --min-timestep "$MIN_TIMESTEP"
    --max-timestep "$MAX_TIMESTEP"
    --prompt-template "$PROMPT_TEMPLATE"
    --min-pair-iou "$MIN_PAIR_IOU"
    --max-pair-iou "$MAX_PAIR_IOU"
    --min-pair-distance "$MIN_PAIR_DISTANCE"
    --max-pair-distance "$MAX_PAIR_DISTANCE"
    --min-view-cos "$MIN_VIEW_COS"
    --min-rotation-deg "$MIN_ROTATION_DEG"
    --max-rotation-deg "$MAX_ROTATION_DEG"
    --max-focal-ratio "$MAX_FOCAL_RATIO"
    --prefilter-topk "$PREFILTER_TOPK"
    --targets-per-ref "$TARGETS_PER_REF"
    --max-pairs-per-scene "$MAX_PAIRS_PER_SCENE"
    --val-ratio "$VAL_RATIO"
    --sample-every-epochs "$SAMPLE_EVERY_EPOCHS"
    --sample-steps "$SAMPLE_STEPS"
    --sample-cfg-scale "$SAMPLE_CFG_SCALE"
    --num-debug-pairs "$NUM_DEBUG_PAIRS"
    --num-debug-samples "$NUM_DEBUG_SAMPLES"
    --sample-seed-base "$SAMPLE_SEED_BASE"
    --H "$H"
    --W "$W"
)

if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
    ARGS+=(--gradient-checkpointing)
fi
if [[ "$XFORMERS_ATTENTION" == "1" ]]; then
    ARGS+=(--xformers-attention)
fi

if [[ -n "$SCENES" ]]; then
    # SCENES should be a space-separated list of scene paths.
    # shellcheck disable=SC2206
    SCENE_LIST=($SCENES)
    ARGS+=(--scenes "${SCENE_LIST[@]}")
elif [[ -n "$SCENES_FILE" ]]; then
    if [[ ! -f "$SCENES_FILE" ]]; then
        echo "Scenes file not found: $SCENES_FILE"
        exit 1
    fi
    ARGS+=(--scenes-file "$SCENES_FILE")
else
    ARGS+=(--scenes "$SCENE")
fi

python -m css.debug_single_ref_experiment "${ARGS[@]}"
