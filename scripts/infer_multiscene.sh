#!/usr/bin/env bash
# Evaluate a multiscene checkpoint on split-held-out targets.

set -euo pipefail

ROOT=${ROOT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling}
TEST_RATIO=${TEST_RATIO:-0.10}
SEED=${SEED:-42}
SPLIT_TAG="test${TEST_RATIO}_seed${SEED}"
SPLIT_TAG=${SPLIT_TAG//./p}

SPLIT_DIR=${SPLIT_DIR:-"$ROOT/splits/multiscene_scenes_${SPLIT_TAG}"}
SCENES_ROOT=${SCENES_ROOT:-"$ROOT"}
OUTPUT=${OUTPUT:-"checkpoints/pose_sd_multiscene_${SPLIT_TAG}"}

if [[ "$OUTPUT" = /* ]]; then
    DEFAULT_CHECKPOINT="$OUTPUT/unet_final.pt"
else
    DEFAULT_CHECKPOINT="$ROOT/$OUTPUT/unet_final.pt"
fi
CHECKPOINT=${CHECKPOINT:-"$DEFAULT_CHECKPOINT"}

OUT_DIR=${OUT_DIR:-"$ROOT/outputs/multiscene_split_eval_$(basename "${CHECKPOINT%.*}")_${SPLIT_TAG}"}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"a photo of {scene}"}
NUM_STEPS=${NUM_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-7.5}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.5}
MIN_DIR_SIM=${MIN_DIR_SIM:-0.2}
MIN_REF_SPACING=${MIN_REF_SPACING:-0.25}
TARGET_IDX=${TARGET_IDX:-all}
MAX_TARGETS_PER_SCENE=${MAX_TARGETS_PER_SCENE:-}
EVAL_SEED=${EVAL_SEED:-42}
H=${H:-512}
W=${W:-512}
START_T=${START_T:-500}
NOISY_TARGET_START=${NOISY_TARGET_START:-0}

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi
cd "$ROOT"

if [[ ! -f "$SPLIT_DIR/test_scenes.txt" ]]; then
    echo "Missing test_scenes.txt in $SPLIT_DIR"
    echo "Run scripts/train.sh first (or set SPLIT_DIR)."
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT"
    echo "Set CHECKPOINT=/path/to/checkpoint.pt"
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "Using checkpoint: $CHECKPOINT"
echo "Split dir: $SPLIT_DIR"
echo "Writing outputs to: $OUT_DIR"

ARGS=(
    --checkpoint "$CHECKPOINT"
    --split-dir "$SPLIT_DIR"
    --output-dir "$OUT_DIR"
    --scenes-root "$SCENES_ROOT"
    --prompt-template "$PROMPT_TEMPLATE"
    --num-steps "$NUM_STEPS"
    --cfg-scale "$CFG_SCALE"
    --max-pair-dist "$MAX_PAIR_DIST"
    --min-dir-sim "$MIN_DIR_SIM"
    --min-ref-spacing "$MIN_REF_SPACING"
    --seed "$EVAL_SEED"
    --H "$H"
    --W "$W"
    --start-t "$START_T"
)

if [[ -n "$MAX_TARGETS_PER_SCENE" ]]; then
    ARGS+=(--max-targets-per-scene "$MAX_TARGETS_PER_SCENE")
fi
if [[ "$TARGET_IDX" == "all" ]]; then
    ARGS+=(--all-targets)
else
    ARGS+=(--target-idx "$TARGET_IDX")
fi
if [[ "$NOISY_TARGET_START" == "1" ]]; then
    ARGS+=(--noisy-target-start)
fi

python -m css.eval_multiscene_split "${ARGS[@]}"
