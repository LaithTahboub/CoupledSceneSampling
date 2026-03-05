#!/bin/bash
# Delete image subfolders with fewer than MIN_IMAGES images,
# remove empty scenes, and update MegaScenes metadata txt files.
#
# A "subfolder" is any directory containing a `pictures/` subdirectory
# under a scene's `images/` tree. The entire parent of `pictures/` is
# the unit that gets deleted (e.g. .../commons/SubScene/0/).
#
# Usage:
#   bash scripts/cleanup_small_subfolders.sh              # dry run (default)
#   bash scripts/cleanup_small_subfolders.sh --execute    # actually delete
#   MIN_IMAGES=60 bash scripts/cleanup_small_subfolders.sh

set -euo pipefail

MEGASCENES="${MEGASCENES:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling/MegaScenes}"
MIN_IMAGES="${MIN_IMAGES:-50}"
DRY_RUN=true

if [[ "${1:-}" == "--execute" ]]; then
    DRY_RUN=false
fi

echo "MegaScenes root: $MEGASCENES"
echo "Min images per subfolder: $MIN_IMAGES"
echo "Dry run: $DRY_RUN"
echo "---"

deleted_subfolders=0
deleted_scenes=0
scenes_to_remove=()

for scene_dir in "$MEGASCENES"/*/; do
    scene_name=$(basename "$scene_dir")
    images_root="$scene_dir/images"

    # Skip non-scene directories (e.g. metadata/)
    if [[ ! -d "$images_root" ]]; then
        continue
    fi

    # Find all pictures/ directories under this scene
    pictures_dirs=()
    while IFS= read -r -d '' pdir; do
        pictures_dirs+=("$pdir")
    done < <(find "$images_root" -type d -name "pictures" -print0 2>/dev/null)

    if (( ${#pictures_dirs[@]} == 0 )); then
        continue
    fi

    has_remaining=false

    for pictures_dir in "${pictures_dirs[@]}"; do
        count=$(find "$pictures_dir" -maxdepth 1 -type f | wc -l)
        # The subfolder to delete is the parent of pictures/ (the .../0/ dir)
        subfolder=$(dirname "$pictures_dir")
        rel_path="${pictures_dir#"$images_root"/}"

        if (( count < MIN_IMAGES )); then
            echo "DELETE subfolder: $scene_name / $rel_path ($count images)"
            deleted_subfolders=$((deleted_subfolders + 1))
            if [[ "$DRY_RUN" == false ]]; then
                rm -rf "$subfolder"
            fi
        else
            has_remaining=true
        fi
    done

    if [[ "$has_remaining" == false ]]; then
        echo "DELETE scene:     $scene_name (no subfolders remaining)"
        deleted_scenes=$((deleted_scenes + 1))
        scenes_to_remove+=("$scene_name")
        if [[ "$DRY_RUN" == false ]]; then
            rm -rf "$scene_dir"
        fi
    fi
done

echo "---"
echo "Subfolders to delete: $deleted_subfolders"
echo "Scenes to delete:     $deleted_scenes"

# Update metadata txt files
txt_files=(
    "$MEGASCENES/scenes_colmap_ready.txt"
    "$MEGASCENES/scenes_colmap_ready_one.txt"
    "$MEGASCENES/scenes_images_only.txt"
    "$MEGASCENES/scenes_images_only_one.txt"
)

if (( ${#scenes_to_remove[@]} > 0 )); then
    echo "---"
    echo "Updating metadata txt files..."

    # Build grep -E pattern: /scene1$|/scene2$|...
    grep_pattern=""
    for s in "${scenes_to_remove[@]}"; do
        if [[ -n "$grep_pattern" ]]; then
            grep_pattern="$grep_pattern|"
        fi
        grep_pattern="$grep_pattern/$s\$"
    done

    for txt in "${txt_files[@]}"; do
        if [[ ! -f "$txt" ]]; then
            continue
        fi

        original_lines=$(wc -l < "$txt")

        if [[ "$DRY_RUN" == false ]]; then
            tmp="$txt.tmp"
            grep -Ev "$grep_pattern" "$txt" > "$tmp" || true
            mv "$tmp" "$txt"
            new_lines=$(wc -l < "$txt")
            echo "  $(basename "$txt"): $original_lines -> $new_lines lines"
        else
            kept=$(grep -Evc "$grep_pattern" "$txt" || true)
            echo "  $(basename "$txt"): $original_lines -> $kept lines (dry run)"
        fi
    done
else
    echo "No scenes fully removed — txt files unchanged."
fi

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "This was a dry run. Re-run with --execute to apply changes."
fi
