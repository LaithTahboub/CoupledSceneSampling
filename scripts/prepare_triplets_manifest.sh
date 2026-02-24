#!/bin/bash
# download scenes one-by-one, keep best-k triplets, and delete raw scenes.

set -euo pipefail

ROOT=${ROOT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling}
OUT_ROOT=${OUT_ROOT:-$ROOT/packed_triplets/megascenes_k3}
CATEGORIES_JSON=${CATEGORIES_JSON:-$ROOT/MegaScenes/metadata/categories.json}
SCENE_LIST_FILE=${SCENE_LIST_FILE:-}
DOWNLOAD_SCRIPT=${DOWNLOAD_SCRIPT:-$ROOT/scripts/grunt/download_scene_data.sh}

TARGET_SCENES=${TARGET_SCENES:-}
MAX_CANDIDATES=${MAX_CANDIDATES:-}
MAX_TRIPLETS_PER_SCENE=${MAX_TRIPLETS_PER_SCENE:-3}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.5}
MIN_DIR_SIM=${MIN_DIR_SIM:-0.2}
MIN_REF_SPACING=${MIN_REF_SPACING:-0.25}
MIN_IMAGES_PER_SCENE=${MIN_IMAGES_PER_SCENE:-15}
SEED=${SEED:-42}
RECON_NO=${RECON_NO:-0}
H=${H:-512}
W=${W:-512}

RESUME=${RESUME:-1}
KEEP_TEMP_SCENES=${KEEP_TEMP_SCENES:-0}
DOWNLOAD_SPARSE=${DOWNLOAD_SPARSE:-1}

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi
cd "$ROOT"

if [[ ! -f "$CATEGORIES_JSON" && -f "$ROOT/categories.json" ]]; then
    CATEGORIES_JSON="$ROOT/categories.json"
fi

if [[ ! -f "$CATEGORIES_JSON" ]]; then
    echo "categories.json not found: $CATEGORIES_JSON"
    exit 1
fi
if [[ ! -f "$DOWNLOAD_SCRIPT" ]]; then
    echo "download script not found: $DOWNLOAD_SCRIPT"
    exit 1
fi

ARGS=(
    --output-root "$OUT_ROOT"
    --categories-json "$CATEGORIES_JSON"
    --download-script "$DOWNLOAD_SCRIPT"
    --target-scenes "$TARGET_SCENES"
    --max-triplets-per-scene "$MAX_TRIPLETS_PER_SCENE"
    --max-pair-dist "$MAX_PAIR_DIST"
    --min-dir-sim "$MIN_DIR_SIM"
    --min-ref-spacing "$MIN_REF_SPACING"
    --min-images-per-scene "$MIN_IMAGES_PER_SCENE"
    --seed "$SEED"
    --recon-no "$RECON_NO"
    --H "$H"
    --W "$W"
)

if [[ -n "$SCENE_LIST_FILE" ]]; then
    ARGS+=(--scene-list-file "$SCENE_LIST_FILE")
fi
if [[ -n "$MAX_CANDIDATES" ]]; then
    ARGS+=(--max-candidates "$MAX_CANDIDATES")
fi
if [[ "$RESUME" == "1" ]]; then
    ARGS+=(--resume)
fi
if [[ "$KEEP_TEMP_SCENES" == "1" ]]; then
    ARGS+=(--keep-temp-scenes)
fi
if [[ "$DOWNLOAD_SPARSE" == "1" ]]; then
    ARGS+=(--download-sparse)
else
    ARGS+=(--no-download-sparse)
fi

export PYTHONUNBUFFERED=1
uv run -m css.prepare_triplet_manifest "${ARGS[@]}"
