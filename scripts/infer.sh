#!/usr/bin/env bash
# Inference from a PoseSD checkpoint.
#
# Modes (set MODE=):
#   triplet  - Single scene + target (default)
#   scenes   - All withheld test scenes from split_info.json
#   targets  - All withheld targets from split_info.json
#
# Examples:
#   MODE=triplet SCENE=/path/to/scene TARGET=image.jpg ./scripts/infer.sh
#   MODE=scenes SPLIT_DIR=splits/pose_sd_seed42 NUM=5 ./scripts/infer.sh
#   MODE=targets SPLIT_DIR=splits/pose_sd_seed42 NUM=10 ./scripts/infer.sh

set -euo pipefail

ROOT=${ROOT:-/vulcanscratch/ltahboub/CoupledSceneSampling}
CHECKPOINT=${CHECKPOINT:-$ROOT/checkpoints/pose_sd_v1/unet_latest.pt}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/pose_sd_seed42}
MODE=${MODE:-triplet}

# Single-triplet mode
SCENE=${SCENE:-}
TARGET=${TARGET:-}  # image name (or empty for random)

# Batch mode: how many to sample
NUM=${NUM:-5}

# Generation params
NUM_STEPS=${NUM_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-4.0}
MIN_COVISIBILITY=${MIN_COVISIBILITY:-0.15}
MAX_COVISIBILITY=${MAX_COVISIBILITY:-0.80}
MIN_DISTANCE=${MIN_DISTANCE:-0.2}
H=${H:-512}
W=${W:-512}
SEED=${SEED:-42}
SHOW_PLUCKERS=${SHOW_PLUCKERS:-0}

OUT_DIR=${OUT_DIR:-$ROOT/outputs}

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi
cd "$ROOT"

[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT"; exit 1; }
mkdir -p "$OUT_DIR"

COMMON_ARGS=(
    --checkpoint "$CHECKPOINT"
    --num-steps "$NUM_STEPS"
    --cfg-scale "$CFG_SCALE"
    --min-covisibility "$MIN_COVISIBILITY"
    --max-covisibility "$MAX_COVISIBILITY"
    --min-distance "$MIN_DISTANCE"
    --H "$H" --W "$W"
    --seed "$SEED"
)
if [[ "$SHOW_PLUCKERS" == "1" ]]; then
    COMMON_ARGS+=(--show-pluckers)
fi

run_one() {
    local scene="$1" target="${2:-}" out_name="$3"
    local args=("${COMMON_ARGS[@]}" --scene "$scene" --output "$OUT_DIR/$out_name")
    if [[ -n "$target" ]]; then
        args+=(--target "$target")
    fi
    echo "=> $out_name"
    python -m css.sample "${args[@]}"
}

case "$MODE" in
triplet)
    if [[ -z "$SCENE" ]]; then
        echo "Set SCENE=/path/to/scene for triplet mode"
        exit 1
    fi
    safe_scene=$(basename "$SCENE" | tr ' ' '_')
    out="triplet_${safe_scene}.png"
    run_one "$SCENE" "$TARGET" "$out"
    ;;

scenes)
    split_json="$SPLIT_DIR/split_info.json"
    [[ -f "$split_json" ]] || { echo "Not found: $split_json"; exit 1; }

    # Read test scenes from split_info.json
    readarray -t test_scenes < <(python3 -c "
import json, sys
info = json.load(open('$split_json'))
for s in info['test_scenes'][:$NUM]:
    print(s)
")
    echo "Withheld scenes: ${#test_scenes[@]} (sampling up to $NUM)"
    for scene_path in "${test_scenes[@]}"; do
        safe=$(basename "$scene_path" | tr ' ' '_')
        run_one "$scene_path" "" "scene_${safe}.png"
    done
    ;;

targets)
    split_json="$SPLIT_DIR/split_info.json"
    [[ -f "$split_json" ]] || { echo "Not found: $split_json"; exit 1; }

    # Read withheld targets: scene_dir<TAB>target_name
    readarray -t entries < <(python3 -c "
import json
info = json.load(open('$split_json'))
wt = info.get('withheld_targets_by_scene', {})
# Need scene dirs — check if train_scenes has full paths
train = {s.split('/')[-1]: s for s in info.get('train_scenes', [])}
count = 0
for scene_name, targets in sorted(wt.items()):
    scene_dir = train.get(scene_name, scene_name)
    for tgt in targets:
        if count >= $NUM:
            break
        print(f'{scene_dir}\t{tgt}')
        count += 1
    if count >= $NUM:
        break
")
    echo "Withheld targets: ${#entries[@]} (sampling up to $NUM)"
    for entry in "${entries[@]}"; do
        scene_dir="${entry%%	*}"
        tgt_name="${entry#*	}"
        safe_scene=$(basename "$scene_dir" | tr ' ' '_')
        safe_tgt=$(basename "$tgt_name" | tr ' .' '_')
        run_one "$scene_dir" "$tgt_name" "target_${safe_scene}_${safe_tgt}.png"
    done
    ;;

*)
    echo "Unknown MODE=$MODE. Use: triplet, scenes, targets"
    exit 1
    ;;
esac
