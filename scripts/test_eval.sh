#!/bin/bash
# Evaluate a split-trained checkpoint on held-out test targets using train-only references.

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
SPLIT_DIR=$ROOT/splits/mysore_palace_test10_seed42

# Override this when needed:
#   CHECKPOINT=/path/to/unet_epoch_100.pt sbatch scripts/test_eval.sh
CHECKPOINT=${CHECKPOINT:-$ROOT/checkpoints/pose_sd_mysore_split_v1/unet_final.pt}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/mysore_palace_split_eval_$(basename "${CHECKPOINT%.*}")}
PROMPT=${PROMPT:-"a photo of the Mysore palace"}
NUM_STEPS=${NUM_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-1.0}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.0}
MIN_DIR_SIM=${MIN_DIR_SIM:-0.3}
MIN_REF_SPACING=${MIN_REF_SPACING:-0.3}

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

# Build the exact target index order used by css.sample (valid scene images filtered by test set, sorted by COLMAP id).
SCENE="$SCENE" TEST_LIST="$SPLIT_DIR/test_images.txt" OUT_DIR="$OUT_DIR" python - <<'PY'
import os
from pathlib import Path

from css.data.colmap_reader import read_scene

scene = Path(os.environ["SCENE"])
test_list = Path(os.environ["TEST_LIST"])
out_dir = Path(os.environ["OUT_DIR"])

with open(test_list, "r", encoding="utf-8") as f:
    test_names = {line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")}

_, images = read_scene(scene)
images_dir = scene / "images"

targets = [img for img in images.values() if (images_dir / img.name).exists() and img.name in test_names]
targets.sort(key=lambda x: x.id)

manifest = out_dir / "test_targets_ordered_by_id.tsv"
with open(manifest, "w", encoding="utf-8") as f:
    f.write("target_idx\timage_id\timage_name\n")
    for i, img in enumerate(targets):
        f.write(f"{i}\t{img.id}\t{img.name}\n")

print(f"Wrote {len(targets)} targets to {manifest}")
PY

NUM_TEST=$(( $(wc -l < "$OUT_DIR/test_targets_ordered_by_id.tsv") - 1 ))
if (( NUM_TEST <= 0 )); then
    echo "No valid test targets found after scene/image filtering."
    exit 1
fi

ok=0
failed=0
for idx in $(seq 0 $((NUM_TEST - 1))); do
    out_file="$OUT_DIR/test_idx_$(printf "%05d" "$idx").png"
    echo "[$((idx + 1))/$NUM_TEST] target_idx=$idx -> $out_file"
    if python -m css.sample \
        --checkpoint "$CHECKPOINT" \
        --scene "$SCENE" \
        --target-idx "$idx" \
        --prompt "$PROMPT" \
        --num-steps "$NUM_STEPS" \
        --cfg-scale "$CFG_SCALE" \
        --max-pair-dist "$MAX_PAIR_DIST" \
        --min-dir-sim "$MIN_DIR_SIM" \
        --min-ref-spacing "$MIN_REF_SPACING" \
        --target-include-image-list "$SPLIT_DIR/test_images.txt" \
        --reference-include-image-list "$SPLIT_DIR/train_images.txt" \
        --output "$out_file"; then
        ok=$((ok + 1))
    else
        failed=$((failed + 1))
        echo "Failed target_idx=$idx"
    fi
done

echo "Eval complete: success=$ok, failed=$failed, total=$NUM_TEST"
if (( failed > 0 )); then
    exit 1
fi
