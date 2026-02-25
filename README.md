# CoupledSceneSampling

Pose-conditioned Stable Diffusion 2.1 baseline for multi-view image synthesis from:
- two reference images,
- reference/target Plucker ray maps,
- text prompt.

This repo is intentionally trimmed to the 2D pipeline only.

## Core files
- `css/models/pose_conditioned_sd.py`: model architecture and sampling.
- `css/train.py`: training loop.
- `css/sample.py`: single-sample inference CLI.
- `css/models/apg.py`: APG guidance.
- `css/models/EMA.py`: EMA + checkpoint I/O.
- `css/data/dataset.py`: dataset + Plucker construction.
- `css/data/colmap_reader.py`: COLMAP binary reader.
- `css/make_scene_split.py`: single-scene train/test image split.
- `css/make_scenes_split.py`: multiscene train/test scene split.
- `css/prepare_triplet_manifest.py`: packed-triplet manifest builder.

## Scripts
- `scripts/grunt/download_scene_data.sh`: download one MegaScenes scene.
- `scripts/grunt/retrieve_scenes.sh`: download many candidate scenes.
- `scripts/grunt/run_colmap_scenes.sh`: run COLMAP when sparse recon is missing.
- `scripts/prepare_multiscene.sh`: end-to-end scene preparation.
- `scripts/prepare_triplets_manifest.sh`: build packed triplets manifest.
- `scripts/train.sh`: multiscene training with automatic split creation.
- `scripts/train_single_scene.sh`: single-scene training with held-out images.
- `scripts/infer.sh`: inference from train/test split index or manual scene.

## Recommended baseline settings
Defaults in `css/train.py`/scripts are tuned for this project direction:
- `--max-pair-dist 2.5`
- `--min-dir-sim 0.2`
- `--min-ref-spacing 0.25`
- `--max-triplets 24`
- `--cond-drop-prob 0.15`
- `--noise-offset 0.05`
- `--min-snr-gamma 5.0`
- `--min-timestep 20 --max-timestep 980`
- `--mixed-precision bf16`

## Quick usage

### 1) Prepare multiscene data
```bash
bash scripts/prepare_multiscene.sh
```

### 2) Train
```bash
bash scripts/train.sh
```

or single-scene:
```bash
SCENE=/path/to/scene bash scripts/train_single_scene.sh
```

### 3) Infer by split index
```bash
CHECKPOINT=/path/to/unet_final.pt \
SPLIT_DIR=/path/to/split_dir \
SPLIT_SET=test \
SPLIT_INDEX=0 \
bash scripts/infer.sh
```
