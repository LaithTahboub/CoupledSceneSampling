import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image, ImageDraw

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
lora_path = "/fs/nexus-scratch/ltahboub/CoupledSceneSampling/mysore_palace_dreambooth_sd3_med/checkpoint-600"

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.load_lora_weights(lora_path)

prompt = "a photo of a landmark at night"

step_options = [40, 60, 80]
guidance_options = [7.5, 12.0, 15.0]

images = []

for steps in step_options:
    for guidance in guidance_options:
        img = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
        ).images[0]
        
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), f"Steps: {steps} | CFG: {guidance}", fill=(255, 255, 255))
        images.append(img)

w, h = images[0].size
cols = len(guidance_options)
rows = len(step_options)

grid = Image.new('RGB', (cols * w, rows * h))

for i, img in enumerate(images):
    grid.paste(img, ((i % cols) * w, (i // cols) * h))

grid.save("palace_grid3.png")