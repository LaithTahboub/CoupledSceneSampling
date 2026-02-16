#!/bin/bash
# Download one MegaScenes scene into <out_root>/<scene_name>/{images,sparse}.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 <scene_id> <scene_name> [recon_no=0] [out_root=../MegaScenes]"
    exit 1
fi

SCENE_ID="$1"
SCENE_NAME="$2"
RECON_NO="${3:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)/MegaScenes"
OUT_ROOT="${4:-${DEFAULT_OUT_ROOT}}"

SCENE_DIR="${OUT_ROOT}/${SCENE_NAME}"
IMAGES_DIR="${SCENE_DIR}/images"
SPARSE_DIR="${SCENE_DIR}/sparse"

mkdir -p "$IMAGES_DIR" "$SPARSE_DIR"

echo "[download] scene=${SCENE_NAME} id=${SCENE_ID} recon=${RECON_NO}"
echo "[download] out=${SCENE_DIR}"

s5cmd --no-sign-request cp "s3://megascenes/images/${SCENE_ID}/*" "${IMAGES_DIR}/"

if s5cmd --no-sign-request cp "s3://megascenes/reconstruct/${SCENE_ID}/colmap/${RECON_NO}/*" "${SPARSE_DIR}/"; then
    echo "[download] used reconstruct/<id>/colmap/${RECON_NO}"
elif s5cmd --no-sign-request cp "s3://megascenes/reconstruct/${SCENE_ID}/sparses/${RECON_NO}/*" "${SPARSE_DIR}/"; then
    echo "[download] used reconstruct/<id>/sparses/${RECON_NO}"
else
    echo "Failed to download sparse COLMAP files for ${SCENE_NAME} (${SCENE_ID})"
    exit 1
fi

if [[ ! -f "${SPARSE_DIR}/cameras.bin" || ! -f "${SPARSE_DIR}/images.bin" ]]; then
    # Some layouts can contain one nested folder under sparse.
    NESTED=$(find "${SPARSE_DIR}" -mindepth 2 -maxdepth 2 -type f -name cameras.bin | head -n 1 || true)
    if [[ -n "${NESTED}" ]]; then
        NESTED_DIR="$(dirname "${NESTED}")"
        cp "${NESTED_DIR}/"*.bin "${SPARSE_DIR}/"
    fi
fi

if [[ ! -f "${SPARSE_DIR}/cameras.bin" || ! -f "${SPARSE_DIR}/images.bin" ]]; then
    echo "Missing cameras.bin/images.bin under ${SPARSE_DIR}"
    exit 1
fi

echo "[ok] downloaded ${SCENE_NAME}"
