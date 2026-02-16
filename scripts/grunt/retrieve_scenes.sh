#!/bin/bash
# Download N MegaScenes scenes (images only) and write scene list.

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

mkdir -p "${OUT_ROOT}" "$(dirname "${CATEGORIES_JSON}")"

if [[ ! -f "${CATEGORIES_JSON}" ]]; then
    echo "[retrieve] downloading categories.json"
    s5cmd --no-sign-request cp "s3://megascenes/metadata/categories.json" "${CATEGORIES_JSON}"
fi

MANIFEST="$(mktemp)"
python3 - "${CATEGORIES_JSON}" "${SCENE_LIST_FILE}" "${COUNT}" > "${MANIFEST}" <<'PY'
import json
import re
import sys
from pathlib import Path

categories_path = Path(sys.argv[1])
scene_list_file = sys.argv[2].strip()
count = int(sys.argv[3])

categories = json.loads(categories_path.read_text(encoding="utf-8"))
if not isinstance(categories, dict):
    raise RuntimeError(f"Unexpected categories format: {categories_path}")

def to_scene_id(value):
    if isinstance(value, dict):
        for k in ("id", "scene_id", "sid", "index"):
            if k in value:
                return to_scene_id(value[k])
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

requested = []
if scene_list_file:
    path = Path(scene_list_file)
    if not path.exists():
        raise RuntimeError(f"Scene list file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if raw and not raw.startswith("#"):
            requested.append(raw)
else:
    requested = sorted(categories.keys())[:count]

selected = []
seen_ids = set()
for raw_name in requested:
    candidate_names = [raw_name, raw_name.replace(" ", "_"), raw_name.replace("_", " ")]
    scene_id = None
    resolved_name = None
    for name in candidate_names:
        if name in categories:
            scene_id = to_scene_id(categories[name])
            if scene_id is not None:
                resolved_name = name
                break
    if scene_id is None or scene_id in seen_ids:
        continue
    seen_ids.add(scene_id)
    safe = f"{scene_id.replace('/', '_')}__{sanitize_scene_name(resolved_name)}"
    selected.append((scene_id, safe, resolved_name))
    if len(selected) >= count:
        break

for scene_id, safe, raw_name in selected:
    print(f"{scene_id}\t{safe}\t{raw_name}")
PY

mkdir -p "$(dirname "${SCENES_OUT_FILE}")"
: > "${SCENES_OUT_FILE}"

ready=0
failed=0
while IFS=$'\t' read -r scene_id scene_safe scene_raw; do
    [[ -n "${scene_id}" && -n "${scene_safe}" ]] || continue
    scene_dir="${OUT_ROOT}/${scene_safe}"
    images_dir="${scene_dir}/images"

    if [[ -d "${images_dir}" ]] && [[ -n "$(find "${images_dir}" -type f -print -quit)" ]]; then
        echo "${scene_dir}" >> "${SCENES_OUT_FILE}"
        ready=$((ready + 1))
        continue
    fi

    if bash "${DOWNLOAD_SCRIPT}" "${scene_id}" "${scene_dir}" "${scene_raw}"; then
        echo "${scene_dir}" >> "${SCENES_OUT_FILE}"
        ready=$((ready + 1))
    else
        echo "[warn] failed ${scene_id} (${scene_raw})"
        failed=$((failed + 1))
    fi
done < "${MANIFEST}"

sort -u "${SCENES_OUT_FILE}" -o "${SCENES_OUT_FILE}"
ready=$(wc -l < "${SCENES_OUT_FILE}" || echo 0)
rm -f "${MANIFEST}"

echo "[retrieve] ready=${ready} failed=${failed}"
echo "[retrieve] scenes list: ${SCENES_OUT_FILE}"
if (( ready < MIN_READY )); then
    echo "[retrieve] expected at least ${MIN_READY} ready scenes"
    exit 1
fi
