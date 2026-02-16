#!/bin/bash
# Run COLMAP reconstruction on downloaded MegaScenes images.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUT_ROOT="${OUT_ROOT:-${ROOT}/MegaScenes}"
SCENES_FILE="${SCENES_FILE:-${OUT_ROOT}/scenes_images_only.txt}"
SCENES_READY_FILE="${SCENES_READY_FILE:-${OUT_ROOT}/scenes_colmap_ready.txt}"
MAX_SCENES="${MAX_SCENES:-100}"

COLMAP_BIN="${COLMAP_BIN:-colmap}"
USE_GPU="${USE_GPU:-1}"
FORCE_RECON="${FORCE_RECON:-0}"
CLEAN_WORK="${CLEAN_WORK:-1}"

SIFT_MAX_FEATURES="${SIFT_MAX_FEATURES:-8192}"
SIFT_MAX_MATCHES="${SIFT_MAX_MATCHES:-32768}"
MAPPER_MAX_NUM_MODELS="${MAPPER_MAX_NUM_MODELS:-1}"
MIN_REGISTERED_IMAGES="${MIN_REGISTERED_IMAGES:-25}"

if [[ ! -f "${SCENES_FILE}" ]]; then
    echo "Scenes file not found: ${SCENES_FILE}"
    exit 1
fi

: > "${SCENES_READY_FILE}"
processed=0
ready=0
failed=0

while IFS= read -r scene_dir; do
    [[ -n "${scene_dir}" ]] || continue
    if (( ready >= MAX_SCENES )); then
        break
    fi
    processed=$((processed + 1))

    images_dir="${scene_dir}/images"
    sparse_dir="${scene_dir}/sparse"
    db_path="${scene_dir}/database.db"
    work_dir="${scene_dir}/colmap_work"

    if [[ ! -d "${images_dir}" ]] || [[ -z "$(find "${images_dir}" -type f -print -quit)" ]]; then
        echo "[skip] no images: ${scene_dir}"
        failed=$((failed + 1))
        continue
    fi

    if [[ "$FORCE_RECON" == "0" && -f "${sparse_dir}/cameras.bin" && -f "${sparse_dir}/images.bin" ]]; then
        echo "[ok] existing sparse: ${scene_dir}"
        echo "${scene_dir}" >> "${SCENES_READY_FILE}"
        ready=$((ready + 1))
        continue
    fi

    rm -rf "${work_dir}"
    mkdir -p "${work_dir}/sparse"
    if [[ "$FORCE_RECON" == "1" ]]; then
        rm -f "${db_path}"
        rm -rf "${sparse_dir}"
    fi

    echo "[colmap] reconstructing ${scene_dir}"
    "${COLMAP_BIN}" feature_extractor \
        --database_path "${db_path}" \
        --image_path "${images_dir}" \
        --SiftExtraction.use_gpu "${USE_GPU}" \
        --SiftExtraction.max_num_features "${SIFT_MAX_FEATURES}" >/dev/null

    if ! "${COLMAP_BIN}" exhaustive_matcher \
        --database_path "${db_path}" \
        --SiftMatching.use_gpu "${USE_GPU}" \
        --SiftMatching.max_num_matches "${SIFT_MAX_MATCHES}" >/dev/null; then
        echo "[warn] exhaustive_matcher failed (gpu=${USE_GPU}) -> retrying on CPU"
        if ! "${COLMAP_BIN}" exhaustive_matcher \
            --database_path "${db_path}" \
            --SiftMatching.use_gpu 0 \
            --SiftMatching.max_num_matches "${SIFT_MAX_MATCHES}" >/dev/null; then
            echo "[fail] matching failed: ${scene_dir}"
            failed=$((failed + 1))
            continue
        fi
    fi

    "${COLMAP_BIN}" mapper \
        --database_path "${db_path}" \
        --image_path "${images_dir}" \
        --output_path "${work_dir}/sparse" \
        --Mapper.max_num_models "${MAPPER_MAX_NUM_MODELS}" >/dev/null || true

    best_model=""
    best_count=-1
    for model_dir in "${work_dir}/sparse"/*; do
        [[ -d "${model_dir}" ]] || continue
        [[ -f "${model_dir}/images.bin" && -f "${model_dir}/cameras.bin" ]] || continue
        count=$(
            "${COLMAP_BIN}" model_analyzer --path "${model_dir}" 2>/dev/null \
            | awk -F': ' '/Registered images/ {print $2; exit}' \
            | tr -dc '0-9'
        )
        if [[ -z "${count}" ]]; then
            count=0
        fi
        if (( count > best_count )); then
            best_count=${count}
            best_model="${model_dir}"
        fi
    done

    if [[ -z "${best_model}" ]]; then
        echo "[fail] no sparse model: ${scene_dir}"
        failed=$((failed + 1))
        continue
    fi

    if (( best_count < MIN_REGISTERED_IMAGES )); then
        echo "[fail] weak reconstruction: ${scene_dir} (registered_images=${best_count})"
        failed=$((failed + 1))
        continue
    fi

    mkdir -p "${sparse_dir}"
    cp "${best_model}/cameras.bin" "${sparse_dir}/"
    cp "${best_model}/images.bin" "${sparse_dir}/"
    if [[ -f "${best_model}/points3D.bin" ]]; then
        cp "${best_model}/points3D.bin" "${sparse_dir}/"
    fi

    if [[ "$CLEAN_WORK" == "1" ]]; then
        rm -rf "${work_dir}"
    fi

    echo "[ok] reconstructed ${scene_dir} (registered_images=${best_count})"
    echo "${scene_dir}" >> "${SCENES_READY_FILE}"
    ready=$((ready + 1))
done < "${SCENES_FILE}"

sort -u "${SCENES_READY_FILE}" -o "${SCENES_READY_FILE}"
ready=$(wc -l < "${SCENES_READY_FILE}" || echo 0)

echo "[colmap] processed=${processed} ready=${ready} failed=${failed}"
echo "[colmap] ready scenes file: ${SCENES_READY_FILE}"
if (( ready < MAX_SCENES )); then
    echo "[colmap] warning: ready scenes (${ready}) < MAX_SCENES (${MAX_SCENES})"
    exit 1
fi
