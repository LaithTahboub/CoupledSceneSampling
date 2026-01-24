import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from PIL import Image, ImageDraw

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model = "stabilityai/stable-diffusion-xl-refiner-1.0"
lora_path = "/fs/nexus-scratch/ltahboub/CoupledSceneSampling/mysore_palace_dreambooth_sdxl/checkpoint-1000"


pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.load_lora_weights(lora_path)


refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    refiner_model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
).to("cuda")

prompt = (
    "A hyperrealistic photo of sks landmark, sunny day, clear blue sky, wide angle."
)


latents = pipe(
    prompt,
    num_inference_steps=50,
    guidance_scale=11,
    output_type="latent",
).images[0]


img = refiner(
    prompt=prompt,
    image=latents[None, :],
).images[0]

img.save("palace_refined.png")
