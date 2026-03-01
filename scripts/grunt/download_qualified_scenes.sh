#!/bin/bash
# Download N new MegaScenes scenes, keeping only subfolders with >= 60 images.
#
# Usage:
#   bash scripts/download_qualified_scenes.sh           # download 50 new scenes (default)
#   COUNT=100 bash scripts/download_qualified_scenes.sh  # download 100 new scenes
#
# Each downloaded scene goes through:
#   1. Download images + precomputed sparse (COLMAP) data
#   2. For each subfolder under images/commons/, count images
#   3. Delete subfolders with < MIN_IMAGES_PER_SUBFOLDER images
#   4. If no subfolders remain, delete the entire scene
#   5. Otherwise, add it to the scenes list

#SBATCH --job-name=download-megascenes
#SBATCH --partition=tron
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=3-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/download_megascenes.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/download_megascenes.err

ROOT="/fs/nexus-scratch/ltahboub/CoupledSceneSampling"
SCRIPT_DIR="${ROOT}/scripts"
DOWNLOAD_SCRIPT="${ROOT}/scripts/grunt/download_scene_data.sh"

OUT_ROOT="${OUT_ROOT:-${ROOT}/MegaScenes}"
COUNT="${COUNT:-150}"
CATEGORIES_JSON="${CATEGORIES_JSON:-${OUT_ROOT}/metadata/categories.json}"
SCENES_READY_FILE="${SCENES_READY_FILE:-${OUT_ROOT}/scenes_colmap_ready.txt}"
MIN_IMAGES_PER_SUBFOLDER="${MIN_IMAGES_PER_SUBFOLDER:-50}"
RECON_NO="${RECON_NO:-0}"
SHUFFLE_SCENES="${SHUFFLE_SCENES:-1}"
RANDOM_SEED="${RANDOM_SEED:-42}"

LOG_DIR="${ROOT}/logs"
LOG_FILE="${LOG_DIR}/download_qualified_scenes.log"
mkdir -p "${OUT_ROOT}" "$(dirname "${CATEGORIES_JSON}")" "${LOG_DIR}"
echo "=== download_qualified_scenes.sh  $(date '+%Y-%m-%d %H:%M:%S')  COUNT=${COUNT}  MIN_IMAGES=${MIN_IMAGES_PER_SUBFOLDER} ===" >> "${LOG_FILE}"

if [[ ! -f "${CATEGORIES_JSON}" ]]; then
    echo "[download] fetching categories.json"
    s5cmd --no-sign-request cp "s3://megascenes/metadata/categories.json" "${CATEGORIES_JSON}"
fi

if [[ ! -f "${DOWNLOAD_SCRIPT}" ]]; then
    echo "Download script not found: ${DOWNLOAD_SCRIPT}"
    exit 1
fi

# Build set of already-downloaded scene IDs (to skip).
EXISTING_IDS="$(mktemp)"
if [[ -f "${SCENES_READY_FILE}" ]]; then
    while IFS= read -r scene_dir; do
        [[ -n "${scene_dir}" ]] || continue
        id_file="${scene_dir}/scene_id.txt"
        if [[ -f "${id_file}" ]]; then
            cat "${id_file}"
        fi
    done < "${SCENES_READY_FILE}" | sort -u > "${EXISTING_IDS}"
else
    : > "${EXISTING_IDS}"
fi
existing_count=$(wc -l < "${EXISTING_IDS}" | tr -d ' ')
echo "[download] ${existing_count} scenes already downloaded"

# Build candidate manifest from categories.json
MANIFEST="$(mktemp)"
python3 - "${CATEGORIES_JSON}" > "${MANIFEST}" <<'PY'
import json, re, sys
from pathlib import Path

categories = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

def to_scene_id(value):
    if isinstance(value, dict):
        for k in ("id", "scene_id", "sid", "index"):
            if k in value:
                return to_scene_id(value[k])
    if isinstance(value, list):
        for x in value:
            sid = to_scene_id(x)
            if sid is not None:
                return sid
        return None
    if isinstance(value, int):
        s = f"{value:06d}"
        return f"{s[:3]}/{s[3:]}"
    if isinstance(value, str):
        s = value.strip()
        if "/" in s:
            return s
        if s.isdigit():
            s = s.zfill(6)
            return f"{s[:3]}/{s[3:]}"
    return None

def sanitize(name):
    s = name.replace("/", "__").replace('"', "").replace("'", "")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._")
    return s or "scene"

for raw_name, raw_value in categories.items():
    sid = to_scene_id(raw_value)
    if sid is None:
        continue
    safe = f"{sid.replace('/', '_')}__{sanitize(raw_name)}"
    print(f"{sid}\t{safe}\t{raw_name}")
PY

if [[ "${SHUFFLE_SCENES}" == "1" ]]; then
    shuffled="$(mktemp)"
    python3 -c "
import random, sys
from pathlib import Path
lines = [x for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()]
random.seed(int(sys.argv[2]))
random.shuffle(lines)
for l in lines: print(l)
" "${MANIFEST}" "${RANDOM_SEED}" > "${shuffled}"
    mv "${shuffled}" "${MANIFEST}"
fi

# Main download loop
ready=0
failed=0
skipped_existing=0
skipped_small=0
skipped_no_recon=0
processed=0

while IFS=$'\t' read -r scene_id scene_safe scene_raw; do
    [[ -n "${scene_id}" && -n "${scene_safe}" ]] || continue
    if (( ready >= COUNT )); then
        break
    fi

    # Skip already downloaded
    if grep -Fxq "${scene_id}" "${EXISTING_IDS}" 2>/dev/null; then
        skipped_existing=$((skipped_existing + 1))
        continue
    fi

    processed=$((processed + 1))

    # Check remote reconstruction exists
    if ! s5cmd --no-sign-request ls "s3://megascenes/reconstruct/${scene_id}/colmap/${RECON_NO}/images.bin" >/dev/null 2>&1; then
        if ! s5cmd --no-sign-request ls "s3://megascenes/reconstruct/${scene_id}/sparses/${RECON_NO}/images.bin" >/dev/null 2>&1; then
            skipped_no_recon=$((skipped_no_recon + 1))
            continue
        fi
    fi

    scene_dir="${OUT_ROOT}/${scene_safe}"

    # Download
    if ! DOWNLOAD_SPARSE=1 RECON_NO="${RECON_NO}" \
        bash "${DOWNLOAD_SCRIPT}" "${scene_id}" "${scene_dir}" "${scene_raw}"; then
        echo "[warn] download failed: ${scene_id} (${scene_raw})"
        echo "FAIL   ${scene_id}  ${scene_safe}  reason=download_failed" >> "${LOG_FILE}"
        failed=$((failed + 1))
        continue
    fi

    images_dir="${scene_dir}/images"
    commons_dir="${images_dir}/commons"

    # Enforce min images per subfolder
    total_subfolders=0
    kept_subfolders=0
    if [[ -d "${commons_dir}" ]]; then
        for subscene_dir in "${commons_dir}"/*/; do
            [[ -d "${subscene_dir}" ]] || continue
            total_subfolders=$((total_subfolders + 1))
            img_count=$(find "${subscene_dir}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.gif' -o -iname '*.bmp' -o -iname '*.tiff' -o -iname '*.webp' \) | wc -l | tr -d ' ')
            if (( img_count < MIN_IMAGES_PER_SUBFOLDER )); then
                rm -rf "${subscene_dir}"
            else
                kept_subfolders=$((kept_subfolders + 1))
            fi
        done
    fi

    # Check if scene still has images
    remaining=$(find "${images_dir}" -type f 2>/dev/null | wc -l | tr -d ' ')
    remaining="${remaining:-0}"
    if (( remaining == 0 )); then
        echo "[skip] ${scene_id}: no subfolders with >= ${MIN_IMAGES_PER_SUBFOLDER} images"
        echo "SKIP   ${scene_id}  ${scene_safe}  subfolders=${kept_subfolders}/${total_subfolders}  images=0  reason=no_qualifying_subfolders" >> "${LOG_FILE}"
        rm -rf "${scene_dir}"
        skipped_small=$((skipped_small + 1))
        continue
    fi

    # Verify sparse data exists
    if [[ ! -f "${scene_dir}/sparse/cameras.bin" || ! -f "${scene_dir}/sparse/images.bin" ]]; then
        echo "[warn] missing COLMAP data after download: ${scene_dir}"
        echo "FAIL   ${scene_id}  ${scene_safe}  subfolders=${kept_subfolders}/${total_subfolders}  images=${remaining}  reason=missing_colmap" >> "${LOG_FILE}"
        rm -rf "${scene_dir}"
        failed=$((failed + 1))
        continue
    fi

    # Add to scenes list
    abs_scene_dir="$(cd "${scene_dir}" && pwd)"
    if ! grep -Fxq "${abs_scene_dir}" "${SCENES_READY_FILE}" 2>/dev/null; then
        echo "${abs_scene_dir}" >> "${SCENES_READY_FILE}"
    fi

    ready=$((ready + 1))
    echo "[ok] ${scene_id} (${scene_raw}): ${remaining} images kept"
    echo "ADMIT  ${scene_id}  ${scene_safe}  subfolders=${kept_subfolders}/${total_subfolders}  images=${remaining}" >> "${LOG_FILE}"
done < "${MANIFEST}"

# Sort and deduplicate the scenes file
sort -u "${SCENES_READY_FILE}" -o "${SCENES_READY_FILE}"
total_scenes=$(wc -l < "${SCENES_READY_FILE}" | tr -d ' ')

rm -f "${MANIFEST}" "${EXISTING_IDS}"

echo "=== SUMMARY: processed=${processed} admitted=${ready} skipped_small=${skipped_small} skipped_no_recon=${skipped_no_recon} failed=${failed} total=${total_scenes} ===" >> "${LOG_FILE}"

echo ""
echo "============================================"
echo "DOWNLOAD SUMMARY"
echo "============================================"
echo "New scenes downloaded: ${ready}"
echo "Failed downloads: ${failed}"
echo "Skipped (already have): ${skipped_existing}"
echo "Skipped (no COLMAP recon): ${skipped_no_recon}"
echo "Skipped (no qualifying subfolders): ${skipped_small}"
echo "Total candidates processed: ${processed}"
echo "Total scenes in ${SCENES_READY_FILE}: ${total_scenes}"
echo "Log file: ${LOG_FILE}"
