#!/bin/bash
#SBATCH --job-name=css-train-multiscene
#SBATCH --partition=tron
#SBATCH --ntasks=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa6000:1
##SBATCH --account=vulcan-jbhuang
##SBATCH --qos=default
#SBATCH --time=3-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_multiscene_split.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_multiscene_split.err

set -euo pipefail
SCRIPT_DIR="/fs/nexus-scratch/ltahboub/CoupledSceneSampling/scripts"
ROOT=${ROOT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling}
MEGASCENES_ROOT=${MEGASCENES_ROOT:-$ROOT/MegaScenes}
SCENES_FILE=${SCENES_FILE:-$MEGASCENES_ROOT/scenes_colmap_ready.txt}

TEST_RATIO=${TEST_RATIO:-0.10}
SEED=${SEED:-42}
MAX_SCENES=${MAX_SCENES:-100}
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
    --max-scenes "$MAX_SCENES" \
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
SAVE_EVERY=4 KEEP_CHECKPOINTS=5 \
    exec "${SCRIPT_DIR}/train_multiscene.sh"
