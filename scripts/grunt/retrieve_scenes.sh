#!/bin/bash
# Download many MegaScenes scenes by name list (or auto-pick from categories.json).
# SCENE_LIST_FILE lines can be either: "<scene_name>" or "<scene_name><TAB><scene_id>".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_scene_data.sh"

OUT_ROOT="${OUT_ROOT:-${ROOT}/MegaScenes}"
CATEGORIES_JSON="${CATEGORIES_JSON:-${OUT_ROOT}/metadata/categories.json}"
SCENE_LIST_FILE="${SCENE_LIST_FILE:-${SCRIPT_DIR}/scene_names_100.txt}"
AUTO_PICK_COUNT="${AUTO_PICK_COUNT:-140}"
RECON_NO="${RECON_NO:-0}"
MIN_SUCCESS="${MIN_SUCCESS:-100}"

mkdir -p "$(dirname "${CATEGORIES_JSON}")"

if [[ ! -f "${CATEGORIES_JSON}" ]]; then
    echo "[retrieve] downloading categories metadata"
    s5cmd --no-sign-request cp "s3://megascenes/metadata/categories.json" "${CATEGORIES_JSON}"
fi

MANIFEST="$(mktemp)"
python3 - "${CATEGORIES_JSON}" "${SCENE_LIST_FILE}" "${AUTO_PICK_COUNT}" > "${MANIFEST}" <<'PY'
import json
import pathlib
import sys

categories_path = pathlib.Path(sys.argv[1])
scene_list_path = pathlib.Path(sys.argv[2])
auto_count = int(sys.argv[3])

categories = json.loads(categories_path.read_text(encoding="utf-8"))
if not isinstance(categories, dict):
    raise RuntimeError(f"Unexpected categories format in {categories_path}")

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
        if not s:
            return None
        if "/" in s:
            return s
        if s.isdigit():
            s = s.zfill(6)
            return f"{s[:3]}/{s[3:]}"
    return None

def resolve_name(raw_name):
    candidates = [
        raw_name,
        raw_name.replace(" ", "_"),
        raw_name.replace("_", " "),
    ]
    for c in candidates:
        if c in categories:
            scene_id = to_scene_id(categories[c])
            if scene_id is not None:
                return c.replace(" ", "_"), scene_id
    return None, None

if auto_count > 0:
    requested = [(k, None) for k in sorted(categories.keys())[:auto_count]]
else:
    if not scene_list_path.exists():
        raise RuntimeError(f"Scene list file not found: {scene_list_path}")
    requested = []
    for line in scene_list_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            if "\t" in s:
                name, sid = s.split("\t", 1)
                requested.append((name.strip(), sid.strip()))
            else:
                requested.append((s, None))

seen = set()
for name, explicit_id in requested:
    if explicit_id:
        scene_name = name.replace(" ", "_")
        scene_id = to_scene_id(explicit_id)
    else:
        scene_name, scene_id = resolve_name(name)
    if scene_name is None:
        print(f"[warn] unresolved scene name: {name}", file=sys.stderr)
        continue
    if scene_id is None:
        print(f"[warn] invalid scene id for {name}: {explicit_id}", file=sys.stderr)
        continue
    if scene_name in seen:
        continue
    seen.add(scene_name)
    print(f"{scene_name}\t{scene_id}")
PY

success=0
failed=0
skipped=0

while IFS=$'\t' read -r scene_name scene_id; do
    [[ -n "${scene_name}" && -n "${scene_id}" ]] || continue

    if [[ -f "${OUT_ROOT}/${scene_name}/sparse/images.bin" && -f "${OUT_ROOT}/${scene_name}/sparse/cameras.bin" ]]; then
        echo "[skip] ${scene_name} already downloaded"
        skipped=$((skipped + 1))
        continue
    fi

    if bash "${DOWNLOAD_SCRIPT}" "${scene_id}" "${scene_name}" "${RECON_NO}" "${OUT_ROOT}"; then
        success=$((success + 1))
    else
        echo "[warn] failed: ${scene_name} (${scene_id})"
        failed=$((failed + 1))
    fi
done < "${MANIFEST}"

rm -f "${MANIFEST}"

echo "[retrieve] success=${success} skipped=${skipped} failed=${failed}"
if (( success + skipped < MIN_SUCCESS )); then
    echo "[retrieve] expected at least ${MIN_SUCCESS} ready scenes"
    exit 1
fi
