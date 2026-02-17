#!/bin/bash
# Download N MegaScenes scenes and write scene lists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_scene_data.sh"

OUT_ROOT="${OUT_ROOT:-${ROOT}/MegaScenes}"
COUNT="${COUNT:-140}"
MIN_READY="${MIN_READY:-100}"
CATEGORIES_JSON="${CATEGORIES_JSON:-${OUT_ROOT}/metadata/categories.json}"
SCENE_LIST_FILE="${SCENE_LIST_FILE:-}"
SCENES_OUT_FILE="${SCENES_OUT_FILE:-${OUT_ROOT}/scenes_images_only.txt}"
SCENES_READY_FILE="${SCENES_READY_FILE:-${OUT_ROOT}/scenes_colmap_ready.txt}"
MIN_IMAGES_PER_SCENE="${MIN_IMAGES_PER_SCENE:-60}"
REQUIRE_PRECOMPUTED_RECON="${REQUIRE_PRECOMPUTED_RECON:-1}"
DOWNLOAD_PRECOMPUTED_SPARSE="${DOWNLOAD_PRECOMPUTED_SPARSE:-1}"
RECON_NO="${RECON_NO:-0}"
SHUFFLE_SCENES="${SHUFFLE_SCENES:-1}"
RANDOM_SEED="${RANDOM_SEED:-42}"

mkdir -p "${OUT_ROOT}" "$(dirname "${CATEGORIES_JSON}")"

if [[ ! -f "${CATEGORIES_JSON}" ]]; then
    echo "[retrieve] downloading categories.json"
    s5cmd --no-sign-request cp "s3://megascenes/metadata/categories.json" "${CATEGORIES_JSON}"
fi

MANIFEST="$(mktemp)"
python3 - "${CATEGORIES_JSON}" "${SCENE_LIST_FILE}" > "${MANIFEST}" <<'PY'
import json
import re
import sys
from pathlib import Path

categories_path = Path(sys.argv[1])
scene_list_file = sys.argv[2].strip()

categories = json.loads(categories_path.read_text(encoding="utf-8"))
if not isinstance(categories, dict):
    raise RuntimeError(f"Unexpected categories format: {categories_path}")

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

def sanitize_scene_name(name: str) -> str:
    s = name.replace("/", "__")
    s = s.replace('"', "").replace("'", "")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._")
    return s or "scene"

def is_scene_id(token: str) -> bool:
    return bool(re.fullmatch(r"\d{3}/\d{3}", token.strip()))

id_to_name = {}
for raw_name, raw_value in categories.items():
    sid = to_scene_id(raw_value)
    if sid is None:
        continue
    id_to_name.setdefault(sid, raw_name)

selected = []
seen = set()

if scene_list_file:
    path = Path(scene_list_file)
    if not path.exists():
        raise RuntimeError(f"Scene list file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue

        sid = None
        raw_name = None
        if "\t" in raw:
            left, right = raw.split("\t", 1)
            if is_scene_id(left):
                sid = left.strip()
                raw_name = right.strip() or id_to_name.get(sid, sid.replace("/", "_"))
        elif is_scene_id(raw):
            sid = raw
            raw_name = id_to_name.get(sid, sid.replace("/", "_"))
        else:
            for name in (raw, raw.replace(" ", "_"), raw.replace("_", " ")):
                if name in categories:
                    sid = to_scene_id(categories[name])
                    if sid is not None:
                        raw_name = name
                        break
        if sid is None or sid in seen:
            continue
        seen.add(sid)
        selected.append((sid, raw_name))
else:
    for sid, raw_name in id_to_name.items():
        if sid in seen:
            continue
        seen.add(sid)
        selected.append((sid, raw_name))

if not selected:
    raise RuntimeError("No candidate scenes found from categories/scene list")

for scene_id, raw_name in selected:
    safe = f"{scene_id.replace('/', '_')}__{sanitize_scene_name(raw_name)}"
    print(f"{scene_id}\t{safe}\t{raw_name}")
PY

if [[ "${SHUFFLE_SCENES}" == "1" ]]; then
    shuffled="$(mktemp)"
    python3 - "${MANIFEST}" "${RANDOM_SEED}" > "${shuffled}" <<'PY'
import random
import sys
from pathlib import Path
lines = [x for x in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if x.strip()]
random.seed(int(sys.argv[2]))
random.shuffle(lines)
for line in lines:
    print(line)
PY
    mv "${shuffled}" "${MANIFEST}"
fi

mkdir -p "$(dirname "${SCENES_OUT_FILE}")"
: > "${SCENES_OUT_FILE}"
if [[ "${DOWNLOAD_PRECOMPUTED_SPARSE}" == "1" ]]; then
    mkdir -p "$(dirname "${SCENES_READY_FILE}")"
    : > "${SCENES_READY_FILE}"
fi

ready=0
ready_with_sparse=0
failed=0
skipped_small=0
skipped_no_recon=0
while IFS=$'\t' read -r scene_id scene_safe scene_raw; do
    [[ -n "${scene_id}" && -n "${scene_safe}" ]] || continue
    if (( ready >= COUNT )); then
        break
    fi

    scene_dir="${OUT_ROOT}/${scene_safe}"
    images_dir="${scene_dir}/images"
    sparse_dir="${scene_dir}/sparse"
    image_count=0
    has_sparse=0
    need_download=0

    if [[ -d "${images_dir}" ]]; then
        image_count=$(find "${images_dir}" -type f | wc -l | tr -d ' ')
        image_count="${image_count:-0}"
    fi
    if [[ -f "${sparse_dir}/cameras.bin" && -f "${sparse_dir}/images.bin" ]]; then
        has_sparse=1
    fi

    if (( image_count < MIN_IMAGES_PER_SCENE )); then
        remote_count=$(s5cmd --no-sign-request ls "s3://megascenes/images/${scene_id}/*" 2>/dev/null | wc -l | tr -d ' ')
        remote_count="${remote_count:-0}"
        if (( remote_count < MIN_IMAGES_PER_SCENE )); then
            skipped_small=$((skipped_small + 1))
            continue
        fi
        need_download=1
    fi

    if [[ "${DOWNLOAD_PRECOMPUTED_SPARSE}" == "1" && "${has_sparse}" == "0" ]]; then
        need_download=1
    fi

    if [[ "${REQUIRE_PRECOMPUTED_RECON}" == "1" ]]; then
        if ! s5cmd --no-sign-request ls "s3://megascenes/reconstruct/${scene_id}/colmap/${RECON_NO}/images.bin" >/dev/null 2>&1; then
            if ! s5cmd --no-sign-request ls "s3://megascenes/reconstruct/${scene_id}/sparses/${RECON_NO}/images.bin" >/dev/null 2>&1; then
                skipped_no_recon=$((skipped_no_recon + 1))
                continue
            fi
        fi
    fi

    if (( need_download == 1 )); then
        if ! DOWNLOAD_SPARSE="${DOWNLOAD_PRECOMPUTED_SPARSE}" RECON_NO="${RECON_NO}" \
            bash "${DOWNLOAD_SCRIPT}" "${scene_id}" "${scene_dir}" "${scene_raw}"; then
            echo "[warn] failed ${scene_id} (${scene_raw})"
            failed=$((failed + 1))
            continue
        fi
    fi

    image_count=$(find "${images_dir}" -type f | wc -l | tr -d ' ')
    image_count="${image_count:-0}"
    if [[ -f "${sparse_dir}/cameras.bin" && -f "${sparse_dir}/images.bin" ]]; then
        has_sparse=1
    else
        has_sparse=0
    fi
    if (( image_count < MIN_IMAGES_PER_SCENE )); then
        skipped_small=$((skipped_small + 1))
        continue
    fi
    if [[ "${DOWNLOAD_PRECOMPUTED_SPARSE}" == "1" && "${has_sparse}" == "0" ]]; then
        echo "[warn] missing sparse after selection: ${scene_dir}"
        failed=$((failed + 1))
        continue
    fi

    if ! grep -Fxq "${scene_dir}" "${SCENES_OUT_FILE}" 2>/dev/null; then
        echo "${scene_dir}" >> "${SCENES_OUT_FILE}"
        ready=$((ready + 1))
    fi
    if [[ "${DOWNLOAD_PRECOMPUTED_SPARSE}" == "1" && "${has_sparse}" == "1" ]]; then
        if ! grep -Fxq "${scene_dir}" "${SCENES_READY_FILE}" 2>/dev/null; then
            echo "${scene_dir}" >> "${SCENES_READY_FILE}"
            ready_with_sparse=$((ready_with_sparse + 1))
        fi
    fi
done < "${MANIFEST}"

sort -u "${SCENES_OUT_FILE}" -o "${SCENES_OUT_FILE}"
ready=$(wc -l < "${SCENES_OUT_FILE}" || echo 0)
if [[ "${DOWNLOAD_PRECOMPUTED_SPARSE}" == "1" ]]; then
    sort -u "${SCENES_READY_FILE}" -o "${SCENES_READY_FILE}"
    ready_with_sparse=$(wc -l < "${SCENES_READY_FILE}" || echo 0)
fi
rm -f "${MANIFEST}"

echo "[retrieve] ready_images=${ready} ready_sparse=${ready_with_sparse} failed=${failed} skipped_small=${skipped_small} skipped_no_recon=${skipped_no_recon}"
echo "[retrieve] filters: min_images=${MIN_IMAGES_PER_SCENE} require_precomputed_recon=${REQUIRE_PRECOMPUTED_RECON} download_precomputed_sparse=${DOWNLOAD_PRECOMPUTED_SPARSE} recon_no=${RECON_NO}"
echo "[retrieve] scenes list: ${SCENES_OUT_FILE}"
if [[ "${DOWNLOAD_PRECOMPUTED_SPARSE}" == "1" ]]; then
    echo "[retrieve] sparse-ready list: ${SCENES_READY_FILE}"
    if (( ready_with_sparse < MIN_READY )); then
        echo "[retrieve] expected at least ${MIN_READY} sparse-ready scenes"
        exit 1
    fi
else
    if (( ready < MIN_READY )); then
        echo "[retrieve] expected at least ${MIN_READY} ready scenes"
        exit 1
    fi
fi
