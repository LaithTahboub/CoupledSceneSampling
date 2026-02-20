#!/bin/bash
# Evaluate held-out test targets (split index) using train-only references.

#SBATCH --job-name=css-eval-mysore-split
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=1-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/test_eval.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/test_eval.err

set -euo pipefail

ROOT=/fs/nexus-scratch/ltahboub/CoupledSceneSampling
SCENE=MegaScenes/Mysore_Palace
TEST_RATIO=${TEST_RATIO:-0.10}
TRAIN_RATIO=${TRAIN_RATIO:-1.0}
SEED=${SEED:-42}
SPLIT_TAG="test${TEST_RATIO}_train${TRAIN_RATIO}_seed${SEED}"
SPLIT_TAG=${SPLIT_TAG//./p}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/mysore_palace_${SPLIT_TAG}}

OUTPUT=${OUTPUT:-checkpoints/pose_sd_mysore_${SPLIT_TAG}}
if [[ "$OUTPUT" = /* ]]; then
    DEFAULT_CHECKPOINT="$OUTPUT/unet_final.pt"
else
    DEFAULT_CHECKPOINT="$ROOT/$OUTPUT/unet_final.pt"
fi
CHECKPOINT=${CHECKPOINT:-$DEFAULT_CHECKPOINT}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/mysore_palace_split_eval_$(basename "${CHECKPOINT%.*}")}
PROMPT=${PROMPT:-"a photo of the Mysore palace"}
NUM_STEPS=${NUM_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-7.5}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.0}
MIN_DIR_SIM=${MIN_DIR_SIM:-0.3}
MIN_REF_SPACING=${MIN_REF_SPACING:-0.3}
TARGET_IDX=${TARGET_IDX:-all}
H=${H:-512}
W=${W:-512}
START_T=${START_T:-500}
NOISY_TARGET_START=${NOISY_TARGET_START:-0}

source $ROOT/.venv/bin/activate
cd $ROOT

if [[ ! -f "$SPLIT_DIR/train_images.txt" || ! -f "$SPLIT_DIR/test_images.txt" ]]; then
    echo "Missing split files in $SPLIT_DIR"
    echo "Run scripts/train_split.sh first (or set SPLIT_DIR to an existing split)."
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT"
    echo "Set CHECKPOINT=/path/to/checkpoint.pt when submitting."
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "Using checkpoint: $CHECKPOINT"
echo "Writing outputs to: $OUT_DIR"

ARGS=(
    --checkpoint "$CHECKPOINT"
    --scene "$SCENE"
    --split-dir "$SPLIT_DIR"
    --output-dir "$OUT_DIR"
    --prompt "$PROMPT"
    --num-steps "$NUM_STEPS"
    --cfg-scale "$CFG_SCALE"
    --max-pair-dist "$MAX_PAIR_DIST"
    --min-dir-sim "$MIN_DIR_SIM"
    --min-ref-spacing "$MIN_REF_SPACING"
    --H "$H"
    --W "$W"
    --start-t "$START_T"
)

if [[ "$TARGET_IDX" == "all" ]]; then
    ARGS+=(--all-targets)
else
    ARGS+=(--target-idx "$TARGET_IDX")
fi

if [[ "$NOISY_TARGET_START" == "1" ]]; then
    ARGS+=(--noisy-target-start)
fi

python -m css.eval_split "${ARGS[@]}"
