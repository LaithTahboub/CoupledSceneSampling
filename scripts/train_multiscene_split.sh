#!/bin/bash
# Train on many scenes with per-scene held-out targets and scene-specific prompts.

#SBATCH --job-name=css-train-multiscene
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=4
#SBATCH --mem=64gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=4-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_multiscene.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_multiscene.err

set -euo pipefail

ROOT=${ROOT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling}
MEGASCENES_ROOT=${MEGASCENES_ROOT:-$ROOT/MegaScenes}
SCENES_FILE=${SCENES_FILE:-$ROOT/scene_lists/megascenes_downloaded.txt}
REFRESH_SCENES_FILE=${REFRESH_SCENES_FILE:-1}
MIN_SCENES=${MIN_SCENES:-100}
POSE_QUALITY_FILTER=${POSE_QUALITY_FILTER:-1}
QUALITY_MAX_ROT_ORTH_ERR=${QUALITY_MAX_ROT_ORTH_ERR:-0.05}
QUALITY_MAX_DET_ERR=${QUALITY_MAX_DET_ERR:-0.10}
QUALITY_MIN_POSE_VALID_RATIO=${QUALITY_MIN_POSE_VALID_RATIO:-0.95}
QUALITY_MIN_COVERED_TARGETS=${QUALITY_MIN_COVERED_TARGETS:-20}
QUALITY_MIN_COVERAGE=${QUALITY_MIN_COVERAGE:-0.35}

TEST_RATIO=${TEST_RATIO:-0.10}
TRAIN_RATIO=${TRAIN_RATIO:-1.0}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-100}

PROMPT=${PROMPT:-"a photo of a scene"}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"a photo of {scene}"}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.5}
MIN_DIR_SIM=${MIN_DIR_SIM:-0.2}
MIN_REF_SPACING=${MIN_REF_SPACING:-0.25}
MAX_TRIPLETS=${MAX_TRIPLETS:-3000}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-5e-6}
SAVE_EVERY=${SAVE_EVERY:-10}
COND_DROP_PROB=${COND_DROP_PROB:-0.1}
SAMPLE_CFG_SCALE=${SAMPLE_CFG_SCALE:-1.0}
SAMPLE_APG_ETA=${SAMPLE_APG_ETA:-0.0}
SAMPLE_APG_MOMENTUM=${SAMPLE_APG_MOMENTUM:--0.5}
SAMPLE_APG_NORM_THRESHOLD=${SAMPLE_APG_NORM_THRESHOLD:-0.0}
H=${H:-512}
W=${W:-512}

SPLIT_TAG="multiscene_test${TEST_RATIO}_train${TRAIN_RATIO}_seed${SEED}"
SPLIT_TAG=${SPLIT_TAG//./p}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/${SPLIT_TAG}}
OUTPUT=${OUTPUT:-checkpoints/pose_sd_${SPLIT_TAG}}
RUN_NAME=${RUN_NAME:-css-${SPLIT_TAG}}

source $ROOT/.venv/bin/activate
cd $ROOT

if [[ "$REFRESH_SCENES_FILE" == "1" || ! -f "$SCENES_FILE" ]]; then
    mkdir -p "$(dirname "$SCENES_FILE")"
    while IFS= read -r cameras_bin; do
        sparse_dir="$(dirname "$cameras_bin")"
        scene_dir="$(dirname "$sparse_dir")"
        [[ -d "$scene_dir/images" ]] || continue
        [[ -f "$sparse_dir/images.bin" ]] || continue
        echo "$scene_dir"
    done < <(find "$MEGASCENES_ROOT" -type f -path "*/sparse/cameras.bin" | sort) > "$SCENES_FILE"
    sort -u "$SCENES_FILE" -o "$SCENES_FILE"
    echo "Generated valid scene list: $SCENES_FILE"
fi

READY_COUNT=$(wc -l < "$SCENES_FILE" || echo 0)
echo "Valid scenes found: $READY_COUNT"

python -m css.make_multiscene_split \
    --scenes-file "$SCENES_FILE" \
    --output-dir "$SPLIT_DIR" \
    --test-ratio "$TEST_RATIO" \
    --train-ratio "$TRAIN_RATIO" \
    --seed "$SEED" \
    --min-scenes "$MIN_SCENES" \
    --quality-max-pair-dist "$MAX_PAIR_DIST" \
    --quality-min-dir-sim "$MIN_DIR_SIM" \
    --quality-min-ref-spacing "$MIN_REF_SPACING" \
    --quality-max-rot-orth-err "$QUALITY_MAX_ROT_ORTH_ERR" \
    --quality-max-det-err "$QUALITY_MAX_DET_ERR" \
    --quality-min-pose-valid-ratio "$QUALITY_MIN_POSE_VALID_RATIO" \
    --quality-min-covered-targets "$QUALITY_MIN_COVERED_TARGETS" \
    --quality-min-coverage "$QUALITY_MIN_COVERAGE" \
    $([[ "$POSE_QUALITY_FILTER" == "1" ]] && echo "--pose-quality-filter")

python -m css.train \
    --scenes-file "$SPLIT_DIR/scenes.txt" \
    --output "$OUTPUT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --lr "$LR" \
    --max-pair-dist "$MAX_PAIR_DIST" \
    --min-dir-sim "$MIN_DIR_SIM" \
    --min-ref-spacing "$MIN_REF_SPACING" \
    --max-triplets "$MAX_TRIPLETS" \
    --save-every "$SAVE_EVERY" \
    --prompt "$PROMPT" \
    --prompt-template "$PROMPT_TEMPLATE" \
    --unet-train-mode cond \
    --cond-drop-prob "$COND_DROP_PROB" \
    --sample-cfg-scale "$SAMPLE_CFG_SCALE" \
    --sample-apg-eta "$SAMPLE_APG_ETA" \
    --sample-apg-momentum "$SAMPLE_APG_MOMENTUM" \
    --sample-apg-norm-threshold "$SAMPLE_APG_NORM_THRESHOLD" \
    --min-timestep 0 \
    --exclude-image-list "$SPLIT_DIR/test_images.txt" \
    --target-include-image-list "$SPLIT_DIR/train_images.txt" \
    --reference-include-image-list "$SPLIT_DIR/train_images.txt" \
    --H "$H" \
    --W "$W" \
    --wandb-project CoupledSceneSampling \
    --wandb-name "$RUN_NAME"
