import torch
from diffusers import StableDiffusionPipeline

model_id = (
    "/fs/nexus-scratch/ltahboub/CoupledSceneSampling/mysore_palace_dreambooth_sd21"
)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda"
)

prompt = "A photo of sks palace at night"
image = pipe(prompt, num_inference_steps=100, guidance_scale=10).images[0]

image.save("palace.png")
