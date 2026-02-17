#!/bin/bash
# Prepare multi-scene training data: download images then run COLMAP.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_ROOT="${OUT_ROOT:-${ROOT}/MegaScenes}"
DOWNLOAD_COUNT="${DOWNLOAD_COUNT:-140}"
TARGET_SCENES="${TARGET_SCENES:-100}"
SCENES_IMAGES_FILE="${SCENES_IMAGES_FILE:-${OUT_ROOT}/scenes_images_only.txt}"
SCENES_COLMAP_FILE="${SCENES_COLMAP_FILE:-${OUT_ROOT}/scenes_colmap_ready.txt}"
USE_PRECOMPUTED_RECON="${USE_PRECOMPUTED_RECON:-1}"
MIN_IMAGES_PER_SCENE="${MIN_IMAGES_PER_SCENE:-60}"
REQUIRE_PRECOMPUTED_RECON="${REQUIRE_PRECOMPUTED_RECON:-1}"
DOWNLOAD_PRECOMPUTED_SPARSE="${DOWNLOAD_PRECOMPUTED_SPARSE:-${USE_PRECOMPUTED_RECON}}"
RECON_NO="${RECON_NO:-0}"

COUNT="${DOWNLOAD_COUNT}" MIN_READY="${TARGET_SCENES}" OUT_ROOT="${OUT_ROOT}" \
SCENES_OUT_FILE="${SCENES_IMAGES_FILE}" SCENES_READY_FILE="${SCENES_COLMAP_FILE}" \
MIN_IMAGES_PER_SCENE="${MIN_IMAGES_PER_SCENE}" REQUIRE_PRECOMPUTED_RECON="${REQUIRE_PRECOMPUTED_RECON}" \
DOWNLOAD_PRECOMPUTED_SPARSE="${DOWNLOAD_PRECOMPUTED_SPARSE}" RECON_NO="${RECON_NO}" \
    bash "${ROOT}/scripts/grunt/retrieve_scenes.sh"

if [[ "${USE_PRECOMPUTED_RECON}" == "1" ]]; then
    echo "[prepare] using precomputed sparse reconstructions; skipping local COLMAP."
else
    OUT_ROOT="${OUT_ROOT}" SCENES_FILE="${SCENES_IMAGES_FILE}" SCENES_READY_FILE="${SCENES_COLMAP_FILE}" MAX_SCENES="${TARGET_SCENES}" \
        bash "${ROOT}/scripts/grunt/run_colmap_scenes.sh"
fi

echo "[prepare] done"
echo "[prepare] images list: ${SCENES_IMAGES_FILE}"
echo "[prepare] colmap-ready list: ${SCENES_COLMAP_FILE}"
