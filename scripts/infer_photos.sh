#!/bin/bash
# Infer on arbitrary photos using MAST3R for pose estimation + PoseSD.
#
# Usage:
#   REF1=photo1.jpg REF2=photo2.jpg TARGET=photo3.jpg sbatch scripts/infer_photos.sh
#   REF1=photo1.jpg REF2=photo2.jpg TARGET=photo3.jpg PROMPT="A photo of a cathedral, at sunset" bash scripts/infer_photos.sh

#SBATCH --job-name=css-infer-photos
#SBATCH --partition=tron
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=1-0:00:00
#SBATCH --output=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/infer_photos_%j.out
#SBATCH --error=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/infer_photos_%j.err

set -euo pipefail

ROOT=${ROOT:-/vulcanscratch/ltahboub/CoupledSceneSampling}
CHECKPOINT=${CHECKPOINT:-$ROOT/checkpoints/pose_sd_v9_512x512/unet_latest.pt}

REF1=${REF1:-}
REF2=${REF2:-}
TARGET=${TARGET:-}
PROMPT=${PROMPT:-}

H=${H:-512}
W=${W:-512}
NUM_STEPS=${NUM_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-3.0}
CFG_TEXT=${CFG_TEXT:-3.0}
SEED=${SEED:-42}
ARCH=${ARCH:-NEW}
OUTPUT=${OUTPUT:-$ROOT/outputs/photos_output.png}

if [[ -z "$REF1" ]] || [[ -z "$REF2" ]] || [[ -z "$TARGET" ]]; then
    echo "Set REF1, REF2, and TARGET"
    exit 1
fi

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi
cd "$ROOT"

ARGS=(
    --ref1 "$REF1" --ref2 "$REF2" --target "$TARGET"
    --checkpoint "$CHECKPOINT"
    --H "$H" --W "$W"
    --num-steps "$NUM_STEPS"
    --cfg-scale "$CFG_SCALE"
    --cfg-text "$CFG_TEXT"
    --seed "$SEED"
    --arch "$ARCH"
    --output "$OUTPUT"
)
if [[ -n "$PROMPT" ]]; then
    ARGS+=(--prompt "$PROMPT")
fi

python -m css.inference.infer_triplet "${ARGS[@]}"
