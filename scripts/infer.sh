#!/usr/bin/env bash
# Inference from a RelightSD checkpoint.
#
# Modes (set MODE=):
#   triplet  - Single scene + target (default), auto-select refs
#   images   - Specific ref1, ref2, target within a scene
#   photos   - Arbitrary photos (no COLMAP), DUSt3R for poses
#   scenes   - All withheld test scenes from split_info.json
#   targets  - All withheld targets from split_info.json
#
# Examples:
#   MODE=triplet SCENE=/path/to/scene TARGET=image.jpg ./scripts/infer.sh
#   MODE=images SCENE=/path/to/scene REF1=img1.jpg REF2=img2.jpg TARGET=img3.jpg ./scripts/infer.sh
#   MODE=photos REF1=photo1.jpg REF2=photo2.jpg TARGET=photo3.jpg ./scripts/infer.sh
#   MODE=photos REF1=photo1.jpg REF2=photo2.jpg DIRECTION=right DISTANCE=0.3 ./scripts/infer.sh
#   MODE=scenes SPLIT_DIR=splits/relight_sd_seed42 NUM=5 ./scripts/infer.sh
#   MODE=targets SPLIT_DIR=splits/relight_sd_seed42 NUM=10 ./scripts/infer.sh

set -euo pipefail

ROOT=${ROOT:-/vulcanscratch/ltahboub/CoupledSceneSampling}
CHECKPOINT=${CHECKPOINT:-$ROOT/checkpoints/relight_sd_v1/unet_latest.pt}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/relight_sd_seed42}
DATA_ROOT=${DATA_ROOT:-/fs/nexus-scratch/ltahboub/MegaScenes}
MODE=${MODE:-triplet}

# Single-triplet / images mode
SCENE=${SCENE:-}
TARGET=${TARGET:-}  # image name (or empty for random)
REF1=${REF1:-}      # ref1 image name (images mode)
REF2=${REF2:-}      # ref2 image name (images mode)

# Captions
CAPTION_DIR=${CAPTION_DIR:-/fs/nexus-scratch/ltahboub/MegaScenesCaptions}

# Photos mode (DUSt3R)
DIRECTION=${DIRECTION:-right}
DISTANCE=${DISTANCE:-0.3}
ANCHOR=${ANCHOR:-ref1}
PROMPT=${PROMPT:-}

# Batch mode: how many to sample, starting from TARGET_IDX
NUM=${NUM:-5}
TARGET_IDX=${TARGET_IDX:-0}  # 0-based offset into withheld targets list

# Generation params
NUM_STEPS=${NUM_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-3.0}
CFG_TEXT=${CFG_TEXT:-3.0}
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
    --cfg-text "$CFG_TEXT"
    --min-covisibility "$MIN_COVISIBILITY"
    --max-covisibility "$MAX_COVISIBILITY"
    --min-distance "$MIN_DISTANCE"
    --H "$H" --W "$W"
    --seed "$SEED"
    --caption-dir "$CAPTION_DIR"
)
if [[ -n "$PROMPT" ]]; then
    COMMON_ARGS+=(--prompt "$PROMPT")
fi
if [[ "$SHOW_PLUCKERS" == "1" ]]; then
    COMMON_ARGS+=(--show-pluckers)
fi

run_one() {
    local scene="$1" target="${2:-}" out_name="$3"
    shift 3
    local args=("${COMMON_ARGS[@]}" --scene "$scene" --output "$OUT_DIR/$out_name" "$@")
    if [[ -n "$target" ]]; then
        args+=(--target "$target")
    fi
    echo "=> $out_name"
    python -m css.inference.sample "${args[@]}"
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

images)
    if [[ -z "$SCENE" ]] || [[ -z "$REF1" ]] || [[ -z "$REF2" ]] || [[ -z "$TARGET" ]]; then
        echo "Set SCENE, REF1, REF2, TARGET for images mode"
        exit 1
    fi
    safe_scene=$(basename "$SCENE" | tr ' ' '_')
    safe_tgt=$(basename "$TARGET" | tr ' .' '_')
    out="images_${safe_scene}_${safe_tgt}.png"
    run_one "$SCENE" "$TARGET" "$out" --ref1 "$REF1" --ref2 "$REF2"
    ;;

photos)
    if [[ -z "$REF1" ]] || [[ -z "$REF2" ]]; then
        echo "Set REF1 and REF2 for photos mode"
        exit 1
    fi
    PHOTO_ARGS=(
        --ref1 "$REF1" --ref2 "$REF2"
        --checkpoint "$CHECKPOINT"
        --num-steps "$NUM_STEPS"
        --cfg-scale "$CFG_SCALE"
        --cfg-text "$CFG_TEXT"
        --H "$H" --W "$W"
        --seed "$SEED"
    )
    if [[ -n "$PROMPT" ]]; then
        PHOTO_ARGS+=(--prompt "$PROMPT")
    fi
    if [[ -n "$TARGET" ]]; then
        PHOTO_ARGS+=(--target "$TARGET")
        safe_tgt=$(basename "$TARGET" | tr ' .' '_')
        out="photos_${safe_tgt}.png"
    else
        PHOTO_ARGS+=(--direction "$DIRECTION" --distance "$DISTANCE" --anchor "$ANCHOR")
        out="photos_${DIRECTION}_${DISTANCE}.png"
    fi
    if [[ "$SHOW_PLUCKERS" == "1" ]]; then
        PHOTO_ARGS+=(--show-pluckers)
    fi
    PHOTO_ARGS+=(--output "$OUT_DIR/$out")
    echo "=> $out"
    python -m css.infer_photos "${PHOTO_ARGS[@]}"
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
    for scene_name in "${test_scenes[@]}"; do
        safe=$(basename "$scene_name" | tr ' ' '_')
        run_one "$DATA_ROOT/$scene_name" "" "scene_${safe}.png"
    done
    ;;

targets)
    split_json="$SPLIT_DIR/split_info.json"
    [[ -f "$split_json" ]] || { echo "Not found: $split_json"; exit 1; }

    # Read withheld targets: scene_dir<TAB>target_name (skip TARGET_IDX, take NUM)
    readarray -t entries < <(python3 -c "
import json
info = json.load(open('$split_json'))
wt = info.get('withheld_targets_by_scene', {})
train = {s.split('/')[-1]: s for s in info.get('train_scenes', [])}
all_entries = []
for scene_name, targets in sorted(wt.items()):
    scene_dir = train.get(scene_name, scene_name)
    for tgt in targets:
        all_entries.append(f'{scene_dir}\t{tgt}')
for e in all_entries[$TARGET_IDX:$TARGET_IDX+$NUM]:
    print(e)
")
    echo "Withheld targets: ${#entries[@]} (idx=$TARGET_IDX, num=$NUM)"
    for entry in "${entries[@]}"; do
        scene_dir="${entry%%	*}"
        tgt_name="${entry#*	}"
        safe_scene=$(basename "$scene_dir" | tr ' ' '_')
        safe_tgt=$(basename "$tgt_name" | tr ' .' '_')
        run_one "$DATA_ROOT/$scene_dir" "$tgt_name" "target_${safe_scene}_${safe_tgt}.png" || echo "SKIPPED (failed): $tgt_name"
    done
    ;;

*)
    echo "Unknown MODE=$MODE. Use: triplet, images, photos, scenes, targets"
    exit 1
    ;;
esac
