#!/usr/bin/env bash
# Infer a sample from a checkpoint by train/test split index.

set -euo pipefail
#/fs/nexus-scratch/ltahboub/CoupledSceneSampling/splits/multiscene_scenes_test0p10_seed9
ROOT=${ROOT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling}
SCENES_ROOT=${SCENES_ROOT:-$ROOT}
CHECKPOINT=${CHECKPOINT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling/checkpoints/pose_sd_multiscene_test0p10_seed9/unet_final.pt}
SPLIT_DIR=${SPLIT_DIR:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling/splits/multiscene_scenes_test0p10_seed9}
SPLIT_SET=${SPLIT_SET:-train}          # train | test
SPLIT_INDEX=${SPLIT_INDEX:-}         # 0-based index into split file
TARGET_IDX=${TARGET_IDX:-1}           # target index inside selected scene (scene-split mode)
SCENE=${SCENE:-}
REF_SPLIT_SET=${REF_SPLIT_SET:-auto}  # auto | train | test | same

PROMPT=${PROMPT:-""}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"a photo of {scene}"}
NUM_STEPS=${NUM_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-3.5}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.5}
MIN_PAIR_IOU=${MIN_PAIR_IOU:-0.15}
MIN_REF_SPACING=${MIN_REF_SPACING:-0.25}
H=${H:-512}
W=${W:-512}
START_T=${START_T:-500}
NOISY_TARGET_START=${NOISY_TARGET_START:-0}
SHOW_PLUCKERS=${SHOW_PLUCKERS:-0}
SEED=${SEED:-4}

OUT_DIR=${OUT_DIR:-$ROOT/outputs}
OUT_NAME=${OUT_NAME:-}

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi
cd "$ROOT"

if [[ -z "$CHECKPOINT" ]]; then
    echo "Set CHECKPOINT=/path/to/unet_*.pt"
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [[ -n "$SPLIT_DIR" && ! -d "$SPLIT_DIR" ]]; then
    echo "Split dir not found: $SPLIT_DIR"
    exit 1
fi
if [[ "$SPLIT_SET" != "train" && "$SPLIT_SET" != "test" ]]; then
    echo "SPLIT_SET must be 'train' or 'test'"
    exit 1
fi

mkdir -p "$OUT_DIR"

# Detect split mode.
image_split_file=""
scene_split_file=""
if [[ -n "$SPLIT_DIR" && -f "$SPLIT_DIR/${SPLIT_SET}_images.txt" ]]; then
    image_split_file="$SPLIT_DIR/${SPLIT_SET}_images.txt"
elif [[ -n "$SPLIT_DIR" && -f "$SPLIT_DIR/${SPLIT_SET}_scenes.txt" ]]; then
    scene_split_file="$SPLIT_DIR/${SPLIT_SET}_scenes.txt"
fi

target_filter_tmp=""
cleanup() {
    if [[ -n "$target_filter_tmp" && -f "$target_filter_tmp" ]]; then
        rm -f "$target_filter_tmp"
    fi
}
trap cleanup EXIT

ARGS=(
    --checkpoint "$CHECKPOINT"
    --num-steps "$NUM_STEPS"
    --cfg-scale "$CFG_SCALE"
    --max-pair-dist "$MAX_PAIR_DIST"
    --min-pair-iou "$MIN_PAIR_IOU"
    --min-ref-spacing "$MIN_REF_SPACING"
    --prompt-template "$PROMPT_TEMPLATE"
    --H "$H"
    --W "$W"
    --start-t "$START_T"
    --seed "$SEED"
)

if [[ -n "$PROMPT" ]]; then
    ARGS+=(--prompt "$PROMPT")
fi
if [[ "$NOISY_TARGET_START" == "1" ]]; then
    ARGS+=(--noisy-target-start)
fi
if [[ "$SHOW_PLUCKERS" == "1" ]]; then
    ARGS+=(--show-pluckers)
fi

# Image-split mode: choose exact target image by index.
if [[ -n "$image_split_file" ]]; then
    split_line_no=$((SPLIT_INDEX + 1))
    entry=$(sed -n "${split_line_no}p" "$image_split_file" | tr -d '\r')
    if [[ -z "$entry" ]]; then
        echo "Split index ${SPLIT_INDEX} out of range for $image_split_file"
        exit 1
    fi

    scene_name=""
    image_name="$entry"
    if [[ "$entry" == */* ]]; then
        scene_name="${entry%%/*}"
        image_name="${entry#*/}"
    fi

    if [[ -z "$SCENE" ]]; then
        if [[ -n "$scene_name" ]]; then
            SCENE="$SCENES_ROOT/$scene_name"
        else
            echo "SCENE must be set when split entries do not include scene prefixes"
            exit 1
        fi
    fi

    target_filter_tmp=$(mktemp)
    {
        echo "$image_name"
        if [[ -n "$scene_name" ]]; then
            echo "$scene_name/$image_name"
        fi
    } > "$target_filter_tmp"

    ref_split_set="$REF_SPLIT_SET"
    if [[ "$ref_split_set" == "auto" ]]; then
        if [[ "$SPLIT_SET" == "test" ]]; then
            ref_split_set="train"
        else
            ref_split_set="$SPLIT_SET"
        fi
    elif [[ "$ref_split_set" == "same" ]]; then
        ref_split_set="$SPLIT_SET"
    fi

    ref_split_file="$SPLIT_DIR/${ref_split_set}_images.txt"
    if [[ ! -f "$ref_split_file" ]]; then
        echo "Reference split file not found: $ref_split_file"
        exit 1
    fi

    if [[ -z "$OUT_NAME" ]]; then
        safe_scene=$(basename "$SCENE" | tr ' ' '_')
        safe_image=$(basename "$image_name" | tr ' ' '_')
        ckpt_tag=$(basename "${CHECKPOINT%.*}")
        OUT_NAME="${SPLIT_SET}_${SPLIT_INDEX}_${safe_scene}_${safe_image}_${ckpt_tag}.png"
    fi

    ARGS+=(
        --scene "$SCENE"
        --target-idx 0
        --target-include-image-list "$target_filter_tmp"
        --reference-include-image-list "$ref_split_file"
    )

# Scene-split mode: choose scene by index, then target image by TARGET_IDX.
elif [[ -n "$scene_split_file" ]]; then
    split_line_no=$((SPLIT_INDEX + 1))
    scene_entry=$(sed -n "${split_line_no}p" "$scene_split_file" | tr -d '\r')
    if [[ -z "$scene_entry" ]]; then
        echo "Split index ${SPLIT_INDEX} out of range for $scene_split_file"
        exit 1
    fi

    if [[ -z "$SCENE" ]]; then
        if [[ "$scene_entry" = /* ]]; then
            SCENE="$scene_entry"
        else
            SCENE="$SCENES_ROOT/$scene_entry"
        fi
    fi

    if [[ -z "$OUT_NAME" ]]; then
        safe_scene=$(basename "$SCENE" | tr ' ' '_')
        ckpt_tag=$(basename "${CHECKPOINT%.*}")
        OUT_NAME="${SPLIT_SET}_${SPLIT_INDEX}_${safe_scene}_target${TARGET_IDX}_${ckpt_tag}.png"
    fi

    ARGS+=(
        --scene "$SCENE"
        --target-idx "$TARGET_IDX"
    )

# Manual mode (no split): use SCENE + TARGET_IDX directly.
else
    if [[ -z "$SCENE" ]]; then
        echo "Provide SCENE=/path/to/scene or SPLIT_DIR with split files"
        exit 1
    fi

    if [[ -z "$OUT_NAME" ]]; then
        safe_scene=$(basename "$SCENE" | tr ' ' '_')
        ckpt_tag=$(basename "${CHECKPOINT%.*}")
        OUT_NAME="manual_${safe_scene}_target${TARGET_IDX}_${ckpt_tag}.png"
    fi

    ARGS+=(
        --scene "$SCENE"
        --target-idx "$TARGET_IDX"
    )
fi

OUTPUT_PATH="$OUT_DIR/$OUT_NAME"
ARGS+=(--output "$OUTPUT_PATH")

echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_PATH"
python -m css.sample "${ARGS[@]}"
