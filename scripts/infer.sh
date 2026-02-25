ROOT=/fs/nexus-scratch/ltahboub/CoupledSceneSampling
SPLIT_DIR="$ROOT/splits/multiscene_scenes_test0p10_seed42"
CKPT="/fs/nexus-scratch/ltahboub/CoupledSceneSampling/checkpoints/pose_sd_oldmegscenes_v2/unet_epoch_24.pt"

TRAIN_SCENE_LINE=51
TARGET_IDX=0      

SCENE=$(sed -n "${TRAIN_SCENE_LINE}p" "$SPLIT_DIR/train_scenes.txt")
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
uv run -m css.sample \
  --checkpoint "$CKPT" \
  --scene "$SCENE" \
  --target-idx "$TARGET_IDX" \
  --prompt-template "a photo of {scene}" \
  --output "$ROOT/outputs/train_scene${TRAIN_SCENE_LINE}_target${TARGET_IDX}.png" \
  --cfg-scale 7