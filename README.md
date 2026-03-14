# CoupledSceneSampling

Reconstruct large-scale real-world scenes from in-the-wild images by coupling a 3D diffusion model with a 2D diffusion model that can relight scenes and modify transient objects while preserving geometry by conditioning on nearby reference views.

The 2D model is prepared by fine tuning Stable Diffusion to receive as input two reference images near the target view (not only one because of depth ambiguity), their Plucker ray maps, the Plucker ray map of the target image, and a text prompt.
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
- `--min-covisibility 0.22`
- `--max-covisibility 0.58`
- `--min-distance 0.20`
- `--max-triplets 24`
- `--cond-drop-prob 0.15`
- `--noise-offset 0.05`
- `--min-snr-gamma 5.0`
- `--min-timestep 20 --max-timestep 980`
- `--mixed-precision bf16`

## Quick usage
