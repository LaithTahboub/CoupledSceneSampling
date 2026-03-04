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

ROOT="/fs/nexus-scratch/ltahboub/CoupledSceneSampling"
MEGASCENES_ROOT=${MEGASCENES_ROOT:-$ROOT/MegaScenes}
SCENES_FILE=${SCENES_FILE:-"/fs/nexus-scratch/ltahboub/CoupledSceneSampling/MegaScenes/scenes_colmap_ready.txt"}
SCENES=${SCENES:-}

OUTPUT=${OUTPUT:-$ROOT/checkpoints/single_ref_debug}
SEED=${SEED:-42}

EPOCHS=${EPOCHS:-6}
BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-1e-5}

MIN_PAIR_IOU=${MIN_PAIR_IOU:-0.18}
MAX_PAIR_IOU=${MAX_PAIR_IOU:-0.62}
MIN_PAIR_DISTANCE=${MIN_PAIR_DISTANCE:-0.20}
MAX_PAIR_DISTANCE=${MAX_PAIR_DISTANCE:-2.2}
MIN_VIEW_COS=${MIN_VIEW_COS:-0.80}
MIN_ROTATION_DEG=${MIN_ROTATION_DEG:-3.0}
MAX_ROTATION_DEG=${MAX_ROTATION_DEG:-35.0}

SAMPLE_STEPS=${SAMPLE_STEPS:-40}
SAMPLE_CFG_SCALE=${SAMPLE_CFG_SCALE:-4.0}
NUM_DEBUG_PAIRS=${NUM_DEBUG_PAIRS:-4}
NUM_DEBUG_SAMPLES=${NUM_DEBUG_SAMPLES:-6}

H=${H:-512}
W=${W:-512}

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi
cd "$ROOT"

ARGS=(
    --output "$OUTPUT"
    --seed "$SEED"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --min-pair-iou "$MIN_PAIR_IOU"
    --max-pair-iou "$MAX_PAIR_IOU"
    --min-pair-distance "$MIN_PAIR_DISTANCE"
    --max-pair-distance "$MAX_PAIR_DISTANCE"
    --min-view-cos "$MIN_VIEW_COS"
    --min-rotation-deg "$MIN_ROTATION_DEG"
    --max-rotation-deg "$MAX_ROTATION_DEG"
    --sample-steps "$SAMPLE_STEPS"
    --sample-cfg-scale "$SAMPLE_CFG_SCALE"
    --num-debug-pairs "$NUM_DEBUG_PAIRS"
    --num-debug-samples "$NUM_DEBUG_SAMPLES"
    --H "$H"
    --W "$W"
)

if [[ -n "$SCENES" ]]; then
    # SCENES should be a space-separated list of scene paths.
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

uv run -m css.debug_single_ref_experiment "${ARGS[@]}"
