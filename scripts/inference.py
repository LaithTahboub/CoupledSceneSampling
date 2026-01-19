import torch
from diffusers import StableDiffusionPipeline

model_id = (
    "/fs/nexus-scratch/ltahboub/CoupledSceneSampling/mysore_palace_dreambooth_sd15"
)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda"
)

prompt = "a photo of a sks landmark"
image = pipe(prompt, num_inference_steps=50, guidance_scale=13.0).images[0]

image.save("palace.png")
