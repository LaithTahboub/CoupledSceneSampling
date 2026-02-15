#!/bin/bash
#SBATCH --job-name=css-train-pose-sd-new
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=3-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train.err

set -euo pipefail

ROOT=/fs/nexus-scratch/ltahboub/CoupledSceneSampling
SCENE=MegaScenes/Mysore_Palace
EPOCHS=${EPOCHS:-100}
OUTPUT=${OUTPUT:-checkpoints/pose_sd_mysore_full_v1}
RUN_NAME=${RUN_NAME:-mysore-palace-full-v1}
PROMPT=${PROMPT:-"a photo of the Mysore palace"}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.0}
MAX_TRIPLETS=${MAX_TRIPLETS:-1000000}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-1e-5}
SAVE_EVERY=${SAVE_EVERY:-20}
COND_DROP_PROB=${COND_DROP_PROB:-0.1}
SAMPLE_CFG_SCALE=${SAMPLE_CFG_SCALE:-1.0}
SAMPLE_APG_ETA=${SAMPLE_APG_ETA:-0.0}
SAMPLE_APG_MOMENTUM=${SAMPLE_APG_MOMENTUM:--0.5}
SAMPLE_APG_NORM_THRESHOLD=${SAMPLE_APG_NORM_THRESHOLD:-0.0}
H=${H:-512}
W=${W:-512}

source $ROOT/.venv/bin/activate

cd $ROOT

python -m css.train \
    --scenes "$SCENE" \
    --output "$OUTPUT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --max-pair-dist "$MAX_PAIR_DIST" \
    --max-triplets "$MAX_TRIPLETS" \
    --save-every "$SAVE_EVERY" \
    --prompt "$PROMPT" \
    --unet-train-mode cond \
    --cond-drop-prob "$COND_DROP_PROB" \
    --sample-cfg-scale "$SAMPLE_CFG_SCALE" \
    --sample-apg-eta "$SAMPLE_APG_ETA" \
    --sample-apg-momentum "$SAMPLE_APG_MOMENTUM" \
    --sample-apg-norm-threshold "$SAMPLE_APG_NORM_THRESHOLD" \
    --min-timestep 0 \
    --H "$H" \
    --W "$W" \
    --wandb-project CoupledSceneSampling \
    --wandb-name "$RUN_NAME"
