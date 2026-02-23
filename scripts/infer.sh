ROOT=/fs/nexus-scratch/ltahboub/CoupledSceneSampling
SPLIT_DIR="$ROOT/splits/multiscene_scenes_test0p10_seed42"
CKPT="$ROOT/checkpoints/pose_sd_multiscene_test0p10_seed42/unet_epoch_12.pt"

TRAIN_SCENE_LINE=1 
TARGET_IDX=0      

SCENE=$(sed -n "${TRAIN_SCENE_LINE}p" "$SPLIT_DIR/train_scenes.txt")
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
uv run -m css.sample \
  --checkpoint "$CKPT" \
  --scene "$SCENE" \
  --target-idx "$TARGET_IDX" \
  --prompt-template "a photo of {scene}" \
  --output "$ROOT/outputs/train_scene${TRAIN_SCENE_LINE}_target${TARGET_IDX}.png"
  --cfg-scale 7.5