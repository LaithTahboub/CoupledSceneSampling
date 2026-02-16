#!/bin/bash
# Evaluate held-out targets across multi-scene split using train-only refs.

#SBATCH --job-name=css-eval-multiscene
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=1-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/test_eval_multiscene.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/test_eval_multiscene.err

set -euo pipefail

ROOT=${ROOT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling}
TEST_RATIO=${TEST_RATIO:-0.10}
TRAIN_RATIO=${TRAIN_RATIO:-1.0}
SEED=${SEED:-42}
SPLIT_TAG="multiscene_test${TEST_RATIO}_train${TRAIN_RATIO}_seed${SEED}"
SPLIT_TAG=${SPLIT_TAG//./p}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/${SPLIT_TAG}}

OUTPUT=${OUTPUT:-checkpoints/pose_sd_${SPLIT_TAG}}
if [[ "$OUTPUT" = /* ]]; then
    DEFAULT_CHECKPOINT="$OUTPUT/unet_final.pt"
else
    DEFAULT_CHECKPOINT="$ROOT/$OUTPUT/unet_final.pt"
fi
CHECKPOINT=${CHECKPOINT:-$DEFAULT_CHECKPOINT}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/${SPLIT_TAG}_eval_$(basename "${CHECKPOINT%.*}")}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"a photo of {scene}"}
NUM_STEPS=${NUM_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-1.0}
APG_ETA=${APG_ETA:-0.0}
APG_MOMENTUM=${APG_MOMENTUM:--0.5}
APG_NORM_THRESHOLD=${APG_NORM_THRESHOLD:-0.0}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.5}
MIN_DIR_SIM=${MIN_DIR_SIM:-0.2}
MIN_REF_SPACING=${MIN_REF_SPACING:-0.25}
TARGET_IDX=${TARGET_IDX:-all}
H=${H:-512}
W=${W:-512}
START_T=${START_T:-500}
NOISY_TARGET_START=${NOISY_TARGET_START:-0}

source $ROOT/.venv/bin/activate
cd $ROOT

for f in "$SPLIT_DIR/scenes.txt" "$SPLIT_DIR/train_images.txt" "$SPLIT_DIR/test_images.txt"; do
    if [[ ! -f "$f" ]]; then
        echo "Missing split file: $f"
        exit 1
    fi
done

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "Using checkpoint: $CHECKPOINT"
echo "Writing outputs to: $OUT_DIR"

ARGS=(
    --checkpoint "$CHECKPOINT"
    --split-dir "$SPLIT_DIR"
    --output-dir "$OUT_DIR"
    --prompt-template "$PROMPT_TEMPLATE"
    --num-steps "$NUM_STEPS"
    --cfg-scale "$CFG_SCALE"
    --apg-eta "$APG_ETA"
    --apg-momentum "$APG_MOMENTUM"
    --apg-norm-threshold "$APG_NORM_THRESHOLD"
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

python -m css.eval_multiscene_split "${ARGS[@]}"
