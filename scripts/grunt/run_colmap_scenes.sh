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
MAPPER_MIN_NUM_MATCHES="${MAPPER_MIN_NUM_MATCHES:-8}"
MAPPER_INIT_MIN_NUM_INLIERS="${MAPPER_INIT_MIN_NUM_INLIERS:-50}"
MAPPER_ABS_POSE_MIN_NUM_INLIERS="${MAPPER_ABS_POSE_MIN_NUM_INLIERS:-20}"
MIN_INPUT_IMAGES="${MIN_INPUT_IMAGES:-40}"
MIN_REGISTERED_IMAGES="${MIN_REGISTERED_IMAGES:-25}"

if [[ ! -f "${SCENES_FILE}" ]]; then
    echo "Scenes file not found: ${SCENES_FILE}"
    exit 1
fi

pick_best_model() {
    local sparse_root="$1"
    local best_model=""
    local best_count=-1
    local model_dir
    local count

    for model_dir in "${sparse_root}"/*; do
        [[ -d "${model_dir}" ]] || continue
        [[ -f "${model_dir}/images.bin" && -f "${model_dir}/cameras.bin" ]] || continue
        count=$(
            python3 - "${model_dir}/images.bin" <<'PY'
import struct
import sys
path = sys.argv[1]
try:
    with open(path, "rb") as f:
        header = f.read(8)
    if len(header) != 8:
        print(0)
    else:
        print(struct.unpack("<Q", header)[0])
except Exception:
    print(0)
PY
        )
        if (( count > best_count )); then
            best_count=${count}
            best_model="${model_dir}"
        fi
    done

    if [[ -z "${best_model}" ]]; then
        return 1
    fi
    printf "%s\t%s\n" "${best_model}" "${best_count}"
}

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
    input_count=0

    if [[ ! -d "${images_dir}" ]] || [[ -z "$(find "${images_dir}" -type f -print -quit)" ]]; then
        echo "[skip] no images: ${scene_dir}"
        failed=$((failed + 1))
        continue
    fi
    input_count=$(find "${images_dir}" -type f | wc -l | tr -d ' ')
    input_count="${input_count:-0}"
    if (( input_count < MIN_INPUT_IMAGES )); then
        echo "[skip] too few images: ${scene_dir} (images=${input_count}, min=${MIN_INPUT_IMAGES})"
        failed=$((failed + 1))
        continue
    fi

    if [[ "$FORCE_RECON" == "0" && -f "${sparse_dir}/cameras.bin" && -f "${sparse_dir}/images.bin" ]]; then
        echo "[ok] existing sparse: ${scene_dir}"
        echo "${scene_dir}" >> "${SCENES_READY_FILE}"
        ready=$((ready + 1))
        continue
    fi

    if [[ "$FORCE_RECON" == "0" && -d "${work_dir}/sparse" ]]; then
        if best_line=$(pick_best_model "${work_dir}/sparse"); then
            best_model="$(echo "${best_line}" | cut -f1)"
            best_count="$(echo "${best_line}" | cut -f2)"
            if (( best_count >= MIN_REGISTERED_IMAGES )); then
                mkdir -p "${sparse_dir}"
                cp "${best_model}/cameras.bin" "${sparse_dir}/"
                cp "${best_model}/images.bin" "${sparse_dir}/"
                if [[ -f "${best_model}/points3D.bin" ]]; then
                    cp "${best_model}/points3D.bin" "${sparse_dir}/"
                fi
                echo "[ok] reused existing model ${scene_dir} (registered_images=${best_count})"
                echo "${scene_dir}" >> "${SCENES_READY_FILE}"
                ready=$((ready + 1))
                continue
            fi
        fi
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
        --Mapper.max_num_models "${MAPPER_MAX_NUM_MODELS}" \
        --Mapper.min_num_matches "${MAPPER_MIN_NUM_MATCHES}" \
        --Mapper.init_min_num_inliers "${MAPPER_INIT_MIN_NUM_INLIERS}" \
        --Mapper.abs_pose_min_num_inliers "${MAPPER_ABS_POSE_MIN_NUM_INLIERS}" >/dev/null || true

    if ! best_line=$(pick_best_model "${work_dir}/sparse"); then
        echo "[fail] no sparse model: ${scene_dir}"
        failed=$((failed + 1))
        continue
    fi
    best_model="$(echo "${best_line}" | cut -f1)"
    best_count="$(echo "${best_line}" | cut -f2)"

    if (( best_count < MIN_REGISTERED_IMAGES )); then
        echo "[fail] weak reconstruction: ${scene_dir} (registered_images=${best_count}, input_images=${input_count})"
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

    echo "[ok] reconstructed ${scene_dir} (registered_images=${best_count}, input_images=${input_count})"
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
