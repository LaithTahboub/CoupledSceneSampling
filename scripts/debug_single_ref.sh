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
#SBATCH --output=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/debug_single_ref.out
#SBATCH --error=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/debug_single_ref.err

set -euo pipefail

ROOT="/vulcanscratch/ltahboub/CoupledSceneSampling"
MEGASCENES_ROOT=${MEGASCENES_ROOT:-$ROOT/MegaScenes}
SCENES_FILE=${SCENES_FILE:-"$ROOT/MegaScenes/scenes_colmap_ready.txt"}
SCENES=${SCENES:-}

OUTPUT=${OUTPUT:-$ROOT/checkpoints/single_ref_debug4}
SEED=${SEED:-4}

EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-1e-5}

MIN_COVISIBILITY=${MIN_COVISIBILITY:-0.15}
MAX_COVISIBILITY=${MAX_COVISIBILITY:-0.43}
MAX_PAIRS_PER_SCENE=${MAX_PAIRS_PER_SCENE:-35}

# Split config
TEST_SCENES_PCT=${TEST_SCENES_PCT:-5.0}
TEST_TARGETS_PER_SCENE=${TEST_TARGETS_PER_SCENE:-1}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/single_ref_debug_seed${SEED}}

# Checkpoint config
SAVE_EVERY=${SAVE_EVERY:-7}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-3}

SAMPLE_STEPS=${SAMPLE_STEPS:-50}
SAMPLE_CFG_SCALE=${SAMPLE_CFG_SCALE:-3.5}

H=${H:-512}
W=${W:-512}

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi
cd "$ROOT"

ARGS=(
    --output "$OUTPUT"
    --split-dir "$SPLIT_DIR"
    --seed "$SEED"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --min-covisibility "$MIN_COVISIBILITY"
    --max-covisibility "$MAX_COVISIBILITY"
    --max-pairs-per-scene "$MAX_PAIRS_PER_SCENE"
    --test-scenes-pct "$TEST_SCENES_PCT"
    --test-targets-per-scene "$TEST_TARGETS_PER_SCENE"
    --save-every "$SAVE_EVERY"
    --keep-checkpoints "$KEEP_CHECKPOINTS"
    --sample-steps "$SAMPLE_STEPS"
    --sample-cfg-scale "$SAMPLE_CFG_SCALE"
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
