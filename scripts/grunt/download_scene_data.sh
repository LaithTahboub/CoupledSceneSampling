#!/bin/bash
# Download one MegaScenes scene (images only).

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <scene_id> <scene_dir> [scene_name_raw]"
    exit 1
fi

SCENE_ID="$1"
SCENE_DIR="$2"
SCENE_NAME_RAW="${3:-}"

IMAGES_DIR="${SCENE_DIR}/images"
mkdir -p "${IMAGES_DIR}"

echo "[download] id=${SCENE_ID} -> ${SCENE_DIR}"
s5cmd --no-sign-request cp "s3://megascenes/images/${SCENE_ID}/*" "${IMAGES_DIR}/"

if [[ -n "${SCENE_NAME_RAW}" ]]; then
    printf "%s\n" "${SCENE_NAME_RAW}" > "${SCENE_DIR}/scene_name_raw.txt"
fi
printf "%s\n" "${SCENE_ID}" > "${SCENE_DIR}/scene_id.txt"

if [[ -z "$(find "${IMAGES_DIR}" -type f -print -quit)" ]]; then
    echo "No images downloaded for scene ${SCENE_ID}"
    exit 1
fi

echo "[ok] images downloaded"
