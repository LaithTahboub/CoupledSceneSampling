import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

base_model_path = (
    "/fs/nexus-scratch/ltahboub/CoupledSceneSampling/mysore_palace_dreambooth_sd15"
)
checkpoint_path = "/fs/nexus-scratch/ltahboub/CoupledSceneSampling/mysore_palace_dreambooth_sd15/unet"

unet = UNet2DConditionModel.from_pretrained(checkpoint_path, torch_dtype=torch.float16)

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path, unet=unet, torch_dtype=torch.float16
)

pipe.to("cuda")

prompt = (
    "A hyperrealistic photo of sks palace, sunny day, clear blue sky, wide angle, landscape."
)
image = pipe(prompt, num_inference_steps=50, guidance_scale=11.0).images[0]
image.save("palace.png")
