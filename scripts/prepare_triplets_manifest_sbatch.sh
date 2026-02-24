#!/bin/bash
# sbatch entrypoint for packed triplet preparation with resume support.

#SBATCH --job-name=css-pack-triplets
#SBATCH --partition=tron
#SBATCH --ntasks=4
#SBATCH --mem=64gb
#SBATCH --time=3-0:00:00
#SBATCH --output=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/prepare_triplets_manifest_%j.out
#SBATCH --error=/fs/nexus-scratch/ltahboub/CoupledSceneSampling/logs/prepare_triplets_manifest_%j.err

set -euo pipefail

ROOT=${ROOT:-/fs/nexus-scratch/ltahboub/CoupledSceneSampling}
OUT_ROOT=${OUT_ROOT:-$ROOT/packed_triplets/megascenes_k3}
CATEGORIES_JSON=${CATEGORIES_JSON:-$ROOT/MegaScenes/metadata/categories.json}
SCENE_LIST_FILE=${SCENE_LIST_FILE:-}
TARGET_SCENES=${TARGET_SCENES:-100}
MAX_CANDIDATES=${MAX_CANDIDATES:-}
MAX_TRIPLETS_PER_SCENE=${MAX_TRIPLETS_PER_SCENE:-3}
MAX_PAIR_DIST=${MAX_PAIR_DIST:-2.5}
MIN_DIR_SIM=${MIN_DIR_SIM:-0.2}
MIN_REF_SPACING=${MIN_REF_SPACING:-0.25}
MIN_IMAGES_PER_SCENE=${MIN_IMAGES_PER_SCENE:-60}
SEED=${SEED:-42}
RECON_NO=${RECON_NO:-0}
H=${H:-512}
W=${W:-512}
DOWNLOAD_SPARSE=${DOWNLOAD_SPARSE:-1}
RESUME=${RESUME:-1}
KEEP_TEMP_SCENES=${KEEP_TEMP_SCENES:-0}

# this pipeline uses precomputed sparse data and cpu-side filtering/copying,
# so no gpu reservation is required.

echo "[sbatch] ROOT=$ROOT"
echo "[sbatch] OUT_ROOT=$OUT_ROOT"
echo "[sbatch] TARGET_SCENES=$TARGET_SCENES"
echo "[sbatch] RESUME=$RESUME"

bash "$ROOT/scripts/prepare_triplets_manifest.sh"
