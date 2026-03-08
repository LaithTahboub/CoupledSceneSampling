#!/bin/bash
# PoseSD training: 3-view (ref1, ref2, target) with Plucker ray conditioning.

#SBATCH --job-name=css-pose-sd
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=3-0:00:00
#SBATCH --output=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/train_pose_sd.out
#SBATCH --error=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/train_pose_sd.err

set -euo pipefail

ROOT="/vulcanscratch/ltahboub/CoupledSceneSampling"
SCENES_FILE=${SCENES_FILE:-"$ROOT/MegaScenes/scenes_colmap_ready.txt"}
SCENES=${SCENES:-}

OUTPUT=${OUTPUT:-$ROOT/checkpoints/pose_sd_v1}
SEED=${SEED:-42}

EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-1e-5}
TRAIN_MODE=${TRAIN_MODE:-cond}

MIN_COVISIBILITY=${MIN_COVISIBILITY:-0.15}
MAX_COVISIBILITY=${MAX_COVISIBILITY:-0.48}
MIN_REF_COVISIBILITY=${MIN_REF_COVISIBILITY:-0.20}
MAX_REF_COVISIBILITY=${MAX_REF_COVISIBILITY:-0.65}
MIN_DISTANCE=${MIN_DISTANCE:-0.10}
MAX_TRIPLETS_PER_SCENE=${MAX_TRIPLETS_PER_SCENE:-80}

# Split config
TEST_SCENES_PCT=${TEST_SCENES_PCT:-5.0}
TEST_TARGETS_PER_SCENE=${TEST_TARGETS_PER_SCENE:-1}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/pose_sd_seed${SEED}}

# Checkpoint config
SAVE_EVERY=${SAVE_EVERY:-7}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-5}

SAMPLE_STEPS=${SAMPLE_STEPS:-50}
SAMPLE_CFG_SCALE=${SAMPLE_CFG_SCALE:-3}
COND_DROP_PROB=${COND_DROP_PROB:-0.15}

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
    --train-mode "$TRAIN_MODE"
    --cond-drop-prob "$COND_DROP_PROB"
    --min-covisibility "$MIN_COVISIBILITY"
    --max-covisibility "$MAX_COVISIBILITY"
    --min-ref-covisibility "$MIN_REF_COVISIBILITY"
    --max-ref-covisibility "$MAX_REF_COVISIBILITY"
    --min-distance "$MIN_DISTANCE"
    --max-triplets-per-scene "$MAX_TRIPLETS_PER_SCENE"
    --test-scenes-pct "$TEST_SCENES_PCT"
    --test-targets-per-scene "$TEST_TARGETS_PER_SCENE"
    --save-every "$SAVE_EVERY"
    --keep-checkpoints "$KEEP_CHECKPOINTS"
    --sample-steps "$SAMPLE_STEPS"
    --sample-cfg-scale "$SAMPLE_CFG_SCALE"
    --H "$H"
    --W "$W"
    --gradient-checkpointing
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

uv run -m css.train_pose_sd "${ARGS[@]}"
