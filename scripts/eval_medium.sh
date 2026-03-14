#!/bin/bash
# Evaluate PoseSD on MEDIUM-difficulty triplets from test scenes.
# Generates SAMPLES_PER_TRIPLET outputs per triplet with different seeds.
#
# Usage:
#   sbatch scripts/eval_medium.sh
#   NUM_SCENES=3 TARGETS_PER_SCENE=5 SAMPLES_PER_TRIPLET=3 sbatch scripts/eval_medium.sh

#SBATCH --job-name=css-eval-med
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=1-0:00:00
#SBATCH --output=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/eval_medium_%j.out
#SBATCH --error=/vulcanscratch/ltahboub/CoupledSceneSampling/logs/eval_medium_%j.err

set -euo pipefail

ROOT=/vulcanscratch/ltahboub/CoupledSceneSampling
CHECKPOINT=${CHECKPOINT:-$ROOT/checkpoints/pose_sd_v4/unet_step_160000.pt}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/splits/pose_sd_seed42}
DATA_ROOT=${DATA_ROOT:-$ROOT/MegaScenes}

NUM_SCENES=${NUM_SCENES:-5}
SCENE_IDX=${SCENE_IDX:-16}  # 0-based offset into test scenes list
TARGETS_PER_SCENE=${TARGETS_PER_SCENE:-3}
SAMPLES_PER_TRIPLET=${SAMPLES_PER_TRIPLET:-3}

NUM_STEPS=${NUM_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-3.0}
H=${H:-256}
W=${W:-256}
SEED=${SEED:-12}

OUT_DIR=${OUT_DIR:-$ROOT/outputs/eval_medium_$(date +%Y%m%d_%H%M%S)}

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    source "$ROOT/.venv/bin/activate"
fi
cd "$ROOT"
mkdir -p logs "$OUT_DIR"

[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT"; exit 1; }

echo "=== MEDIUM Difficulty Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo "Split: $SPLIT_DIR"
echo "Scenes: $NUM_SCENES (starting at idx $SCENE_IDX) | Targets/scene: $TARGETS_PER_SCENE | Samples/triplet: $SAMPLES_PER_TRIPLET"
echo "Output: $OUT_DIR"
echo ""

python3 -c "
import json, sys, os, random, traceback
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# Paths from env / shell vars
split_json = '$SPLIT_DIR/split_info.json'
data_root = Path('$DATA_ROOT')
checkpoint = '$CHECKPOINT'
out_dir = Path('$OUT_DIR')
num_scenes = $NUM_SCENES
scene_idx = $SCENE_IDX
targets_per_scene = $TARGETS_PER_SCENE
samples_per_triplet = $SAMPLES_PER_TRIPLET
num_steps = $NUM_STEPS
cfg_scale = $CFG_SCALE
H, W = $H, $W
base_seed = $SEED

np.random.seed(base_seed)
torch.manual_seed(base_seed)
random.seed(base_seed)

# Load split
info = json.load(open(split_json))
test_scenes = info['test_scenes'][scene_idx:scene_idx + num_scenes]
print(f'Test scenes available: {len(info[\"test_scenes\"])}, using idx {scene_idx}..{scene_idx + len(test_scenes) - 1} ({len(test_scenes)} scenes)')

# MEDIUM difficulty covisibility/distance ranges
MEDIUM_MIN_COVIS = 0.25
MEDIUM_MAX_COVIS = 0.50
MEDIUM_MIN_DIST = 0.08
MEDIUM_MAX_DIST = 0.40

# Load model once
from css.models.pose_sd import PoseSD
from css.models.EMA import load_pose_sd_checkpoint
from css.scene_sampling import load_scene_pools, build_single_sample, build_comparison_grid, to_uint8
from css.data.iou import compute_covisibility

print('Loading model...')
model = PoseSD()
load_pose_sd_checkpoint(model, checkpoint, model.device)
model.eval()
print(f'Loaded: {checkpoint}')

total = 0
failed = 0

for scene_name in test_scenes:
    scene_dir = data_root / scene_name
    if not scene_dir.exists():
        print(f'SKIP (not found): {scene_name}')
        continue

    print(f'\n--- Scene: {scene_name} ---')
    try:
        cameras, images_dir, target_images, reference_images = load_scene_pools(scene_dir)
    except Exception as e:
        print(f'  SKIP (load failed): {e}')
        continue

    if len(reference_images) < 2 or len(target_images) == 0:
        print(f'  SKIP (not enough images: {len(target_images)} targets, {len(reference_images)} refs)')
        continue

    # Shuffle targets for variety
    shuffled_targets = list(target_images)
    random.shuffle(shuffled_targets)

    safe_scene = scene_name.replace('/', '_').replace(' ', '_')[:60]
    scene_out = out_dir / safe_scene
    scene_out.mkdir(parents=True, exist_ok=True)

    scene_count = 0
    for target_img in shuffled_targets:
        if scene_count >= targets_per_scene:
            break

        # Find MEDIUM-difficulty reference pairs for this target
        target_pos = target_img.c2w[:3, 3].astype(np.float64)
        candidates = []
        for ref in reference_images:
            if ref.id == target_img.id:
                continue
            dist = float(np.linalg.norm(ref.c2w[:3, 3].astype(np.float64) - target_pos))
            if dist < MEDIUM_MIN_DIST or dist > MEDIUM_MAX_DIST:
                continue
            covis = float(compute_covisibility(target_img, ref))
            if covis < MEDIUM_MIN_COVIS or covis > MEDIUM_MAX_COVIS:
                continue
            candidates.append((covis, dist, ref))

        if len(candidates) < 2:
            continue

        # Pick ref1 = highest covisibility with target, ref2 = next best not too close to ref1
        candidates.sort(key=lambda x: x[0], reverse=True)
        ref1 = candidates[0][2]
        ref2 = None
        for _, _, r in candidates[1:]:
            ref_dist = float(np.linalg.norm(r.c2w[:3, 3] - ref1.c2w[:3, 3]))
            if ref_dist >= MEDIUM_MIN_DIST:
                ref2 = r
                break
        if ref2 is None:
            ref2 = candidates[1][2]

        try:
            sample = build_single_sample(cameras, images_dir, ref1, ref2, target_img, H, W)
            safe_tgt = Path(target_img.name).stem.replace('/', '_').replace(' ', '_')[:40]

            for si in range(samples_per_triplet):
                s = base_seed + si
                with torch.inference_mode():
                    generated = model.sample(
                        ref1_img=sample['ref1_img'],
                        ref2_img=sample['ref2_img'],
                        pl_ref1=sample['plucker_ref1'],
                        pl_ref2=sample['plucker_ref2'],
                        pl_tgt=sample['plucker_tgt'],
                        prompt='',
                        num_steps=num_steps,
                        cfg_scale=cfg_scale,
                        seed=s,
                    )

                grid = build_comparison_grid(
                    sample['ref1_img'][0], sample['ref2_img'][0],
                    sample['target_img'][0], generated[0],
                )

                out_path = scene_out / f'{safe_tgt}_seed{s}.png'
                Image.fromarray(grid).save(out_path)

            print(f'  [{scene_count+1}/{targets_per_scene}] {target_img.name} ({samples_per_triplet} seeds)')
            scene_count += 1
            total += 1

        except Exception as e:
            print(f'  FAILED: {target_img.name}: {e}')
            traceback.print_exc()
            failed += 1

print(f'\n=== Done: {total} triplets x {samples_per_triplet} seeds = {total * samples_per_triplet} images, {failed} failed ===')
print(f'Output: {out_dir}')
"
