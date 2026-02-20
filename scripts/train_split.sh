#!/bin/bash
# Train on a large subset while holding out test images that are never seen in training.

#SBATCH --job-name=css-train-mysore-split
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=3-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_split.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_split.err

set -euo pipefail

ROOT=/fs/nexus-scratch/ltahboub/CoupledSceneSampling
SCENE=MegaScenes/Mysore_Palace
TEST_RATIO=${TEST_RATIO:-0.10}
TRAIN_RATIO=${TRAIN_RATIO:-1.0}
SEED=${SEED:-42}
EPOCHS=${EPOCHS:-100}
PROMPT=${PROMPT:-"a photo of the Mysore palace"}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.0}
MAX_TRIPLETS=${MAX_TRIPLETS:-1000000}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-1e-5}
SAVE_EVERY=${SAVE_EVERY:-4}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-5}
COND_DROP_PROB=${COND_DROP_PROB:-0.1}
SAMPLE_CFG_SCALE=${SAMPLE_CFG_SCALE:-7.5}
H=${H:-512}
W=${W:-512}

SPLIT_TAG="test${TEST_RATIO}_train${TRAIN_RATIO}_seed${SEED}"
SPLIT_TAG=${SPLIT_TAG//./p}
SPLIT_DIR=$ROOT/splits/mysore_palace_${SPLIT_TAG}
OUTPUT=${OUTPUT:-checkpoints/pose_sd_mysore_${SPLIT_TAG}}
RUN_NAME=${RUN_NAME:-mysore-palace-${SPLIT_TAG}}

source $ROOT/.venv/bin/activate
cd $ROOT

python -m css.make_scene_split \
    --scene "$SCENE" \
    --output-dir "$SPLIT_DIR" \
    --test-ratio "$TEST_RATIO" \
    --train-ratio "$TRAIN_RATIO" \
    --seed "$SEED"

python -m css.train \
    --scenes "$SCENE" \
    --output "$OUTPUT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --max-pair-dist "$MAX_PAIR_DIST" \
    --max-triplets "$MAX_TRIPLETS" \
    --save-every "$SAVE_EVERY" \
    --keep-checkpoints "$KEEP_CHECKPOINTS" \
    --prompt "$PROMPT" \
    --unet-train-mode cond \
    --cond-drop-prob "$COND_DROP_PROB" \
    --sample-cfg-scale "$SAMPLE_CFG_SCALE" \
    --min-timestep 0 \
    --exclude-image-list "$SPLIT_DIR/test_images.txt" \
    --target-include-image-list "$SPLIT_DIR/train_images.txt" \
    --reference-include-image-list "$SPLIT_DIR/train_images.txt" \
    --H "$H" \
    --W "$W" \
    --wandb-project CoupledSceneSampling \
    --wandb-name "$RUN_NAME"
