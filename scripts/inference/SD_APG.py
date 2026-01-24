import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from typing import Optional, Union, List, Dict, Any

# --- APG Implementation (Algorithm 1) ---

class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average

def project(v0: torch.Tensor, v1: torch.Tensor):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

def adaptive_projected_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale: float,
    momentum_buffer: Optional[MomentumBuffer] = None,
    eta: float = 0.0,
    norm_threshold: float = 0.0,
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[1, 2, 3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
        
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    
    return pred_guided

# --- Minimal APG Pipeline ---

class APGStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        # APG params
        apg_momentum: float = -0.5,
        apg_eta: float = 0.0,
        apg_norm_threshold: float = 0.0,
        # Boilerplate args
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        original_size: Optional[Dict[str, int]] = None,
        crops_coords_top_left: Optional[Dict[str, int]] = None,
        target_size: Optional[Dict[str, int]] = None,
        clip_skip: Optional[int] = None,
        **kwargs,
    ):
        # 0. Init
        momentum_buffer = MomentumBuffer(apg_momentum) if apg_momentum != 0 else None
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        do_classifier_free_guidance = guidance_scale > 1.0

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        crops_coords_top_left = crops_coords_top_left or (0, 0)

        # 1. Encode Inputs
        (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.encode_prompt(
            prompt=prompt, prompt_2=prompt_2, device=self.device,
            num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            clip_skip=clip_skip
        )

        # 2. Prepare Timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # 3. Prepare Latents
        latents = self.prepare_latents(
            1 * num_images_per_prompt, self.unet.config.in_channels, 
            height, width, prompt_embeds.dtype, self.device, None, latents
        )

        # 4. Prepare Added Time IDs
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, 
            dtype=prompt_embeds.dtype, text_encoder_projection_dim=int(pooled_prompt_embeds.shape[-1])
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
        
        add_time_ids = add_time_ids.to(device=self.device).repeat(1 * num_images_per_prompt, 1)

        # 5. Denoising Loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=prompt_embeds, 
                cross_attention_kwargs=cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs, return_dict=False
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                sigma = self.scheduler.sigmas[i]
                pred_original_sample_uncond = latents - sigma * noise_pred_uncond
                pred_original_sample_text = latents - sigma * noise_pred_text

                pred_original_sample_guided = adaptive_projected_guidance(
                    pred_cond=pred_original_sample_text,
                    pred_uncond=pred_original_sample_uncond,
                    guidance_scale=guidance_scale,
                    momentum_buffer=momentum_buffer,
                    eta=apg_eta,
                    norm_threshold=apg_norm_threshold
                )
                noise_pred = (latents - pred_original_sample_guided) / sigma

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 6. Post-processing (FIX: Force VAE decode in float32 to prevent black image)
        if output_type == "latent":
            return StableDiffusionXLPipelineOutput(images=latents, nsfw_content_detected=None)
            
        # Manually decode VAE in float32
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        # Force float32 for decoding
        self.vae.to(dtype=torch.float32)
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        
        image = self.vae.decode(latents.float() / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        # Restore VAE to original precision (optional, but good practice)
        self.vae.to(dtype=torch.float16)

        if not return_dict:
            return (image, None)
            
        return StableDiffusionXLPipelineOutput(images=image)

# --- Execution ---

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
lora_path = "/fs/nexus-scratch/ltahboub/CoupledSceneSampling/mysore_palace_dreambooth_sdxl/checkpoint-1000"

pipe = APGStableDiffusionXLPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Additional Fix: Enable VAE slicing/tiling to save memory and reduce overflow risk
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(lora_path)

prompt = "a photo of a sks landmark at night"

image = pipe(
    prompt,
    num_inference_steps=100,
    guidance_scale=15, 
    apg_momentum=-0.5, 
    apg_eta=0,
).images[0]

image.save("palace_apg.png")                            