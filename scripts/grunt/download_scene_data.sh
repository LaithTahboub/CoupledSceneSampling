#!/bin/bash
# Download one MegaScenes scene (images; optional precomputed sparse).

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <scene_id> <scene_dir> [scene_name_raw]"
    exit 1
fi

SCENE_ID="$1"
SCENE_DIR="$2"
SCENE_NAME_RAW="${3:-}"

DOWNLOAD_SPARSE="${DOWNLOAD_SPARSE:-0}"
RECON_NO="${RECON_NO:-0}"
SKIP_IMAGE_DOWNLOAD_IF_PRESENT="${SKIP_IMAGE_DOWNLOAD_IF_PRESENT:-1}"

IMAGES_DIR="${SCENE_DIR}/images"
mkdir -p "${IMAGES_DIR}"

echo "[download] id=${SCENE_ID} -> ${SCENE_DIR}"
if [[ "${SKIP_IMAGE_DOWNLOAD_IF_PRESENT}" == "1" && -n "$(find "${IMAGES_DIR}" -type f -print -quit 2>/dev/null)" ]]; then
    echo "[download] images already present; skipping image copy"
else
    s5cmd --no-sign-request cp "s3://megascenes/images/${SCENE_ID}/*" "${IMAGES_DIR}/"
fi

if [[ "${DOWNLOAD_SPARSE}" == "1" ]]; then
    SPARSE_DIR="${SCENE_DIR}/sparse"
    mkdir -p "${SPARSE_DIR}"

    if s5cmd --no-sign-request cp "s3://megascenes/reconstruct/${SCENE_ID}/colmap/${RECON_NO}/*" "${SPARSE_DIR}/"; then
        :
    elif s5cmd --no-sign-request cp "s3://megascenes/reconstruct/${SCENE_ID}/sparses/${RECON_NO}/*" "${SPARSE_DIR}/"; then
        :
    else
        echo "Failed to download sparse COLMAP files for scene ${SCENE_ID} (recon=${RECON_NO})"
        exit 1
    fi

    # Some layouts include one nested folder under sparse/.
    if [[ ! -f "${SPARSE_DIR}/cameras.bin" || ! -f "${SPARSE_DIR}/images.bin" ]]; then
        nested="$(find "${SPARSE_DIR}" -mindepth 2 -maxdepth 2 -type f -name cameras.bin | head -n 1 || true)"
        if [[ -n "${nested}" ]]; then
            nested_dir="$(dirname "${nested}")"
            cp "${nested_dir}/"*.bin "${SPARSE_DIR}/" 2>/dev/null || true
        fi
    fi

    if [[ ! -f "${SPARSE_DIR}/cameras.bin" || ! -f "${SPARSE_DIR}/images.bin" ]]; then
        echo "Missing cameras.bin/images.bin under ${SPARSE_DIR}"
        exit 1
    fi
fi

if [[ -n "${SCENE_NAME_RAW}" ]]; then
    printf "%s\n" "${SCENE_NAME_RAW}" > "${SCENE_DIR}/scene_name_raw.txt"
fi
printf "%s\n" "${SCENE_ID}" > "${SCENE_DIR}/scene_id.txt"

if [[ -z "$(find "${IMAGES_DIR}" -type f -print -quit)" ]]; then
    echo "No images downloaded for scene ${SCENE_ID}"
    exit 1
fi

echo "[ok] images downloaded"
