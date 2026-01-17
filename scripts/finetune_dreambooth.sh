#!/bin/bash
#SBATCH --job-name=finetuning_dreambooth
#SBATCH --partition=vulcan-ampere
#SBATCH --ntasks=1
#SBATCH --mem=48gb
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-medium
#SBATCH --time=24:00:00
#SBATCH --output=finetuning_dreambooth.out
#SBATCH --error=finetuning_dreambooth.err

export MODEL_NAME="Manojb/stable-diffusion-2-1-base"
# export INSTANCE_DIR="/fs/nexus-scratch/ltahboub/CoupledSceneSampling/train_ready_data"
export INSTANCE_DIR="/fs/nexus-scratch/ltahboub/CoupledSceneSampling/test_final/Mysore_Palace/images/commons/East_side_of_the_Mysore_Palace/0/pictures"
export CLASS_DIR="/fs/nexus-scratch/ltahboub/CoupledSceneSampling/class_data"  # <--- Added this directory for generic images
export OUTPUT_DIR="/fs/nexus-scratch/ltahboub/CoupledSceneSampling/mysore_palace_dreambooth_sd21"

# accelerate launch --mixed_precision="fp16" train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --instance_data_dir=$INSTANCE_DIR \
#   --class_data_dir=$CLASS_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="a photo of sks palace" \
#   --resolution=768 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=800 \
#   --train_text_encoder \
#   --mixed_precision="fp16" \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   # --class_prompt="a photo of a palace" \
#   --num_class_images=200 \
#   --report_to="wandb" \
#   --validation_prompt="a photo of sks palace at sunset, cinematic lighting" \
#   --validation_steps=100 \
#   --push_to_hub
#
source /fs/nexus-scratch/ltahboub/CoupledSceneSampling/.venv/bin/activate
accelerate launch /fs/nexus-scratch/ltahboub/CoupledSceneSampling/scripts/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks palace" \
  --resolution=768 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub \
  --report_to="wandb"
