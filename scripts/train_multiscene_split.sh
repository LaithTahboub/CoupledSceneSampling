#!/bin/bash
# Train on a scene-level split (test scenes are fully unseen during training).

#SBATCH --job-name=css-train-multiscene
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=1-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_multiscene.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_multiscene.err

set -euo pipefail
SCRIPT_DIR="/fs/nexus-scratch/ltahboub/CoupledSceneSampling/scripts"
ROOT=${ROOT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling}
MEGASCENES_ROOT=${MEGASCENES_ROOT:-$ROOT/MegaScenes}
SCENES_FILE=${SCENES_FILE:-$MEGASCENES_ROOT/scenes_colmap_ready.txt}

TEST_RATIO=${TEST_RATIO:-0.10}
SEED=${SEED:-42}
MIN_TRAIN_SCENES=${MIN_TRAIN_SCENES:-50}
SPLIT_TAG="test${TEST_RATIO}_seed${SEED}"
SPLIT_TAG=${SPLIT_TAG//./p}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/multiscene_scenes_${SPLIT_TAG}}

source "$ROOT/.venv/bin/activate"
cd "$ROOT"

python -m css.make_scenes_split \
    --scenes-file "$SCENES_FILE" \
    --output-dir "$SPLIT_DIR" \
    --test-ratio "$TEST_RATIO" \
    --seed "$SEED" \
    --min-train-scenes "$MIN_TRAIN_SCENES"

TRAIN_SCENES_FILE="$SPLIT_DIR/train_scenes.txt"
TEST_SCENES_FILE="$SPLIT_DIR/test_scenes.txt"

TRAIN_SCENES_COUNT=$(wc -l < "$TRAIN_SCENES_FILE" || echo 0)
TEST_SCENES_COUNT=$(wc -l < "$TEST_SCENES_FILE" || echo 0)
echo "Train scenes: $TRAIN_SCENES_COUNT"
echo "Test scenes: $TEST_SCENES_COUNT"
echo "Test scenes file: $TEST_SCENES_FILE"

RUN_NAME=${RUN_NAME:-multiscene-${SPLIT_TAG}}
OUTPUT=${OUTPUT:-checkpoints/pose_sd_multiscene_${SPLIT_TAG}}

SCENES_FILE="$TRAIN_SCENES_FILE" RUN_NAME="$RUN_NAME" OUTPUT="$OUTPUT" MIN_SCENES=1 \
    exec "${SCRIPT_DIR}/train_multiscene.sh"
