#!/bin/bash
#SBATCH --job-name=css-toy-overfit
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=12:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_toy.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/train_toy.err

set -euo pipefail

source /fs/nexus-scratch/ltahboub/CoupledSceneSampling/.venv/bin/activate

cd /fs/nexus-scratch/ltahboub/CoupledSceneSampling

SCENE=${SCENE:-MegaScenes/Mysore_Palace}
OUTPUT=${OUTPUT:-checkpoints/pose_sd_toy_overfit_v1}
EPOCHS=${EPOCHS:-500}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-0}
LR=${LR:-5e-5}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.0}
MAX_TRIPLETS=${MAX_TRIPLETS:-8}
SAVE_EVERY=${SAVE_EVERY:-50}
KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-5}
COND_DROP_PROB=${COND_DROP_PROB:-0.0}
SAMPLE_CFG_SCALE=${SAMPLE_CFG_SCALE:-1.0}
MIN_TIMESTEP=${MIN_TIMESTEP:-300}
MAX_TIMESTEP=${MAX_TIMESTEP:-}
WARMUP_STEPS=${WARMUP_STEPS:-500}
EMA_DECAY=${EMA_DECAY:-0.9999}
H=${H:-512}
W=${W:-512}
RUN_NAME=${RUN_NAME:-toy-overfit-v5}
WANDB_ID=${WANDB_ID:-}
PROMPT=${PROMPT:-"a photo of the Mysore palace"}

TRAIN_ARGS=(
    --scenes "$SCENE"
    --output "$OUTPUT"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --lr "$LR"
    --warmup-steps "$WARMUP_STEPS"
    --ema-decay "$EMA_DECAY"
    --max-pair-dist "$MAX_PAIR_DIST"
    --max-triplets "$MAX_TRIPLETS"
    --save-every "$SAVE_EVERY"
    --keep-checkpoints "$KEEP_CHECKPOINTS"
    --cond-drop-prob "$COND_DROP_PROB"
    --sample-cfg-scale "$SAMPLE_CFG_SCALE"
    --min-timestep "$MIN_TIMESTEP"
    --unet-train-mode cond
    --H "$H"
    --W "$W"
    --wandb-project CoupledSceneSampling
    --wandb-name "$RUN_NAME"
    --prompt "$PROMPT"
)

if [[ -n "$MAX_TIMESTEP" ]]; then
    TRAIN_ARGS+=(--max-timestep "$MAX_TIMESTEP")
fi
if [[ -n "$WANDB_ID" ]]; then
    TRAIN_ARGS+=(--wandb-id "$WANDB_ID")
fi

python -m css.train "${TRAIN_ARGS[@]}"
