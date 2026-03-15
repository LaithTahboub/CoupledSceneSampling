import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Optional, Dict
from diffusers import DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

# =============================================================================
# VRAM Management & Mathematical Helpers
# =============================================================================

class SequentialOffloadManager:
    """
    Manages VRAM by sequentially loading and offloading UNets during the 
    diffusion loop to prevent OOM errors on 24GB GPUs.
    """
    def __init__(self, use_offloading: bool = True, device: torch.device = torch.device('cuda')):
        self.use_offloading = use_offloading
        self.device = device

    def load(self, model: torch.nn.Module):
        if self.use_offloading:
            model.to(self.device)

    def offload(self, model: torch.nn.Module):
        if self.use_offloading:
            model.to('cpu')
            torch.cuda.empty_cache()

def compute_x0_hat(epsilon: torch.Tensor, x_t: torch.Tensor, t: int, scheduler: DDPMScheduler) -> torch.Tensor:
    """
    Isolates the clean image estimate \hat{x}_0 from the current noisy latent x_t
    and the predicted noise epsilon, utilizing the DDPM cumulative alpha formulation.
    """
    alpha_prod_t = scheduler.alphas_cumprod[t]
    beta_prod_t = 1 - alpha_prod_t
    
    x0_hat = (x_t - math.sqrt(beta_prod_t) * epsilon) / math.sqrt(alpha_prod_t)
    return x0_hat

def compute_ddpm_step(x0_hat: torch.Tensor, epsilon: torch.Tensor, t: int, scheduler: DDPMScheduler, generator: torch.Generator) -> torch.Tensor:
    """
    Computes the reverse diffusion step x_{t-1} using the stochastic DDPM formulation.
    Stochastic re-injection of noise is critical for coupled manifold relaxation.
    """
    alpha_prod_t_prev = scheduler.alphas_cumprod[t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps] if t > 0 else torch.tensor(1.0)
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = scheduler.alphas_cumprod[t] / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # Re-inject noise
    noise = torch.randn(x0_hat.shape, generator=generator, device=x0_hat.device, dtype=x0_hat.dtype)
    
    # Variance scaling
    variance = (beta_prod_t_prev / (1 - scheduler.alphas_cumprod[t])) * current_beta_t
    sigma_t = math.sqrt(variance) if t > 0 else 0.0

    # DDPM Mean Equation
    pred_mean = (math.sqrt(alpha_prod_t_prev) * current_beta_t / (1 - scheduler.alphas_cumprod[t])) * x0_hat + \
                (math.sqrt(current_alpha_t) * beta_prod_t_prev / (1 - scheduler.alphas_cumprod[t])) * (x0_hat * math.sqrt(scheduler.alphas_cumprod[t]) + epsilon * math.sqrt(1 - scheduler.alphas_cumprod[t]))
    
    x_prev = pred_mean + sigma_t * noise
    return x_prev

# =============================================================================
# Coupled Diffusion Pipeline Architecture
# =============================================================================

class CoupledDiffusionPipeline:
    def __init__(
        self,
        vae: AutoencoderKL,
        unet_2d: torch.nn.Module,      # Pose2D Model
        unet_3d: torch.nn.Module,      # Stable Virtual Camera (SEVA)
        scheduler: DDPMScheduler,      # SD 2.1 Aligned DDPM Scheduler
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        vision_encoder: CLIPVisionModelWithProjection,
        device: torch.device = torch.device('cuda'),
        offload_memory: bool = True
    ):
        self.vae = vae.to(device)
        self.unet_2d = unet_2d
        self.unet_3d = unet_3d
        self.scheduler = scheduler
        self.text_encoder = text_encoder.to(device)
        self.tokenizer = tokenizer
        self.vision_encoder = vision_encoder.to(device)
        self.device = device
        
        self.offloader = SequentialOffloadManager(use_offloading=offload_memory, device=device)

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        text_embeddings = self.text_encoder(inputs.input_ids.to(self.device))
        return text_embeddings

    @torch.no_grad()
    def get_clip_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        clip_inputs = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        clip_emb = self.vision_encoder(clip_inputs).image_embeds
        return clip_emb.unsqueeze(1) #

    def _pack_views_2d(self, target_latent: torch.Tensor, ref_latents: List, plucker_maps: List) -> torch.Tensor:
        """
        Packs inputs into the 11-channel format required by Pose2D's CrossViewAttention.
        Returns: tensor.
        """
        b, c, h, w = target_latent.shape
        dtype = target_latent.dtype

        # Masks: 1 for reference, 0 for target
        m_tgt = torch.zeros((b, 1, h, w), device=self.device, dtype=dtype)
        m_r1 = torch.ones((b, 1, h, w), device=self.device, dtype=dtype)
        m_r2 = torch.ones((b, 1, h, w), device=self.device, dtype=dtype)

        pl_tgt, pl_r1, pl_r2 = plucker_maps

        # Channel 0-3: Latent | 4-9: Plucker | 10: Mask
        v_tgt = torch.cat([target_latent, pl_tgt, m_tgt], dim=1) 
        v_r1 = torch.cat([ref_latents, pl_r1, m_r1], dim=1)
        v_r2 = torch.cat([ref_latents, pl_r2, m_r2], dim=1)

        # Output order aligns with Pose2D training slot logic
        packed = torch.stack([v_tgt, v_r1, v_r2], dim=1)
        return packed.reshape(b * 3, 11, h, w)

    @torch.no_grad()
    def __call__(
        self,
        ref_images: List,       # x 2
        plucker_maps: List,     #
        prompt: str,
        lambda_coupling: float = 0.015,       # Coupling guidance strength
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        
        batch_size = ref_images.shape
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 1. Condition Encoding
        text_emb = self.encode_text(prompt)
        uncond_emb = self.encode_text("")
        text_emb_batched = torch.cat([uncond_emb, text_emb])

        clip_img_emb_r1 = self.get_clip_image_embeddings(ref_images)
        clip_img_emb_r2 = self.get_clip_image_embeddings(ref_images)
        clip_img_embs = torch.cat([clip_img_emb_r1, clip_img_emb_r2], dim=1) 

        # 2. VAE Encoding of References
        ref_latents = [self.vae.encode(ref).latent_dist.sample(generator) * self.vae.config.scaling_factor for ref in ref_images]

        # 3. Initialization of Target Latent
        _, c, h, w = ref_latents.shape
        x_t_target = torch.randn((batch_size, c, h, w), generator=generator, device=self.device, dtype=ref_latents.dtype)
        x_t_target_3d = x_t_target.clone()

        # 4. Main DDPM Denoising Loop
        for i, t in enumerate(tqdm(timesteps, desc="Coupled Diffusion Sampling")):
            
            # Classifier-Free Guidance expansion
            latent_input_2d = torch.cat([x_t_target] * 2)
            latent_input_3d = torch.cat([x_t_target_3d] * 2)
            t_batched = torch.tensor([t] * batch_size * 2, device=self.device)

            # ==========================================
            # STEP A: POSE 2D PREDICTION
            # ==========================================
            self.offloader.load(self.unet_2d)
            
            ref_lats_batched = [torch.cat([r]*2) for r in ref_latents]
            plucker_batched = [torch.cat([pl]*2) for pl in plucker_maps]
            
            packed_2d = self._pack_views_2d(latent_input_2d, ref_lats_batched, plucker_batched)
            
            # Repeat text embeddings across the 3 packed views for CrossViewAttention
            te_repeated = text_emb_batched.repeat_interleave(3, dim=0)
            t_repeated = t_batched.repeat_interleave(3)
            
            noise_pred_2d_packed = self.unet_2d(packed_2d, t_repeated, encoder_hidden_states=te_repeated).sample
            
            # Isolate the target slot prediction
            noise_pred_2d = noise_pred_2d_packed.view(batch_size * 2, 3, c, h, w)[:, 0]
            
            # Apply CFG
            noise_pred_uncond_2d, noise_pred_text_2d = noise_pred_2d.chunk(2)
            noise_pred_2d_final = noise_pred_uncond_2d + guidance_scale * (noise_pred_text_2d - noise_pred_uncond_2d)
            
            self.offloader.offload(self.unet_2d)

            # ==========================================
            # STEP B: SEVA 3D PREDICTION
            # ==========================================
            self.offloader.load(self.unet_3d)
            
            seva_kwargs = {
                "plucker_maps": plucker_batched,
                "clip_embeddings": torch.cat([clip_img_embs]*2),
                "reference_latents": ref_lats_batched
            }
            
            noise_pred_3d_target = self.unet_3d(latent_input_3d, t_batched, encoder_hidden_states=text_emb_batched, **seva_kwargs).sample
            
            # Apply CFG
            noise_pred_uncond_3d, noise_pred_text_3d = noise_pred_3d_target.chunk(2)
            noise_pred_3d_final = noise_pred_uncond_3d + guidance_scale * (noise_pred_text_3d - noise_pred_uncond_3d)
            
            self.offloader.offload(self.unet_3d)

            # ==========================================
            # STEP C: DDPM REVERSE STEP & COUPLING
            # ==========================================
            # Compute independent \hat{x}_0 estimates
            x0_hat_2d = compute_x0_hat(noise_pred_2d_final, x_t_target, t, self.scheduler)
            x0_hat_3d = compute_x0_hat(noise_pred_3d_final, x_t_target_3d, t, self.scheduler)
            
            # Compute base stochastic step x_{t-1}
            x_prev_2d = compute_ddpm_step(x0_hat_2d, noise_pred_2d_final, t, self.scheduler, generator)
            x_prev_3d = compute_ddpm_step(x0_hat_3d, noise_pred_3d_final, t, self.scheduler, generator)

            # Compute scale factor for energy gradient
            alpha_bar_t_prev = self.scheduler.alphas_cumprod[t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps] if t > 0 else 1.0
            scale_factor = math.sqrt(1 - alpha_bar_t_prev)
            
            # Calculate gradient shift
            coupling_grad_2d = lambda_coupling * (x0_hat_2d - x0_hat_3d).detach()
            coupling_grad_3d = lambda_coupling * (x0_hat_3d - x0_hat_2d).detach()

            # Apply trajectory steering
            x_t_target = x_prev_2d - (scale_factor * coupling_grad_2d)
            x_t_target_3d = x_prev_3d - (scale_factor * coupling_grad_3d)

        # 5. Decode final latent
        final_latent = (1 / self.vae.config.scaling_factor) * x_t_target
        image = self.vae.decode(final_latent).sample
        
        return (image / 2 + 0.5).clamp(0, 1)