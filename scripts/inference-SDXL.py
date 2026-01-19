import torch
from diffusers import StableDiffusionXLPipeline

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
# lora_path = "/fs/nexus-scratch/ltahboub/CoupledSceneSampling/mysore_palace_dreambooth_sdxl/checkpoint-3000"

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# pipe.load_lora_weights(lora_path)

prompt = "a photo of a landmark at night"

image = pipe(
    prompt,
    num_inference_steps=50,
    guidance_scale=15,
).images[0]

image.save("palace.png")
