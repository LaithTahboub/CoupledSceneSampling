import os
from argparse import ArgumentParser
import torch
from diffusers import StableDiffusionPipeline, StableVideoDiffusionPipeline
from diffusers.schedulers import EulerDiscreteScheduler
from PIL import Image
import numpy as np
from diffusers.utils import load_image

class CoupledDiffusion:
    def __init__(self, pipe_2d, pipe_mv):
        self.pipe_2d = pipe_2d
        self.pipe_mv = pipe_mv
        
        if str(pipe_2d.device) != str(pipe_mv.device):
            raise ValueError("Pipelines on different devices")

        # ==============================================================================
        # FIX: Use EulerDiscreteScheduler (SVD Native)
        # ==============================================================================
        # 1. Load SVD's native scheduler config.
        # 2. Force the 2D model to use the SAME scheduler config.
        #    This works because SD 2.1 Base is also v-prediction and compatible with Euler.
        
        self.pipe_mv.scheduler = EulerDiscreteScheduler.from_config(
            pipe_mv.scheduler.config
        )
        self.pipe_2d.scheduler = EulerDiscreteScheduler.from_config(
            pipe_mv.scheduler.config
        )
        
    @torch.no_grad()
    def __call__(
        self,
        prompt_2d: str,
        ref_image_mv: Image.Image,
        lambda_: float,
        num_frames: int = 21,
        num_inference_steps: int = 25, # Euler works well with fewer steps (25-30)
        guidance_scale: float = 7.5,
        seed: int = 32,
    ):
        device = self.pipe_2d.device
        dtype = self.pipe_2d.unet.dtype
        generator = torch.Generator(device=device).manual_seed(seed)
        batch_size = 1

        # ==========================================
        # 1. Prepare Conditioning
        # ==========================================
        cond_2d = self._encode_text(self.pipe_2d, prompt_2d, 1, device)
        uncond_2d = self._encode_text(self.pipe_2d, "", 1, device)

        # MV Model Conditioning
        clip_image = self.pipe_mv.feature_extractor(images=ref_image_mv, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
        cond_mv_clip = self.pipe_mv._encode_image(clip_image, device, 1, False)
        uncond_mv_clip = torch.zeros_like(cond_mv_clip)

        # VAE Image Latents (SVD Concatenation)
        vae_image = self.pipe_mv.video_processor.preprocess(ref_image_mv, height=576, width=1024).to(device=device, dtype=dtype)
        image_latents = self.pipe_mv.vae.encode(vae_image).latent_dist.mode()
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        image_latents = image_latents * self.pipe_mv.vae.config.scaling_factor
        image_latents = torch.cat([image_latents] * 2)

        # ==========================================
        # 2. Prepare Noisy Latents
        # ==========================================
        # Use SVD's sigma-based noise scaling
        self.pipe_mv.scheduler.set_timesteps(num_inference_steps, device=device)
        self.pipe_2d.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe_mv.scheduler.timesteps

        # SVD/Euler needs noise scaled by initial sigma
        sigma_init = self.pipe_mv.scheduler.sigmas[0]
        
        latents_2d = torch.randn(
            (1, self.pipe_2d.unet.config.in_channels, 64, 64),
            device=device, dtype=dtype, generator=generator
        ) * sigma_init

        latents_mv = torch.randn(
            (1, num_frames, 4, 72, 128),
            device=device, dtype=dtype, generator=generator
        ) * sigma_init

        # ==========================================
        # 3. Denoising Loop
        # ==========================================
        for i, t in enumerate(timesteps):
            
            # Helper to get current sigma for x0 calculation
            # EulerDiscreteScheduler.sigmas[i] corresponds to time t
            sigma = self.pipe_mv.scheduler.sigmas[i]

            # --- 2D Prediction ---
            # Scale input for Euler
            latent_model_input_2d = torch.cat([latents_2d] * 2)
            latent_model_input_2d = self.pipe_2d.scheduler.scale_model_input(latent_model_input_2d, t)

            noise_pred_2d = self.pipe_2d.unet(
                latent_model_input_2d, t, encoder_hidden_states=torch.cat([uncond_2d, cond_2d])
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_2d.chunk(2)
            noise_pred_2d = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # --- MV Prediction ---
            latent_model_input_mv = torch.cat([latents_mv] * 2)
            latent_model_input_mv = self.pipe_mv.scheduler.scale_model_input(latent_model_input_mv, t)
            latent_model_input_mv = torch.cat([latent_model_input_mv, image_latents], dim=2)
            
            added_time_ids = self.pipe_mv._get_add_time_ids(
                7, 127, 0.02, dtype, batch_size, 1, True
            ).to(device)

            noise_pred_mv = self.pipe_mv.unet(
                latent_model_input_mv, t, 
                encoder_hidden_states=torch.cat([uncond_mv_clip, cond_mv_clip]),
                added_time_ids=added_time_ids
            ).sample

            noise_pred_uncond_mv, noise_pred_cond_mv = noise_pred_mv.chunk(2)
            noise_pred_mv = noise_pred_uncond_mv + 3.0 * (noise_pred_cond_mv - noise_pred_uncond_mv)

            # --- Compute x0 (Original Sample) Manually ---
            # For V-Prediction (which SVD and SD2.1-Base use): x0 = x - sigma * v
            # For Epsilon: x0 = (x - sigma * eps) / alpha (approx)
            # We assume V-prediction here as per config.
            
            # Unscale latents for x0 calculation (remove sigma scaling)
            # step() does this internally, but we need x0 BEFORE the step for coupling? 
            # Actually, standard Algorithm 1 uses the x0 pred from the current step.
            
            # Let's run the step first to get prev_sample
            out_2d = self.pipe_2d.scheduler.step(noise_pred_2d, t, latents_2d, generator=generator)
            latents_2d_prev = out_2d.prev_sample

            latents_mv_flat = latents_mv.flatten(0, 1) 
            noise_pred_mv_flat = noise_pred_mv.flatten(0, 1)
            out_mv = self.pipe_mv.scheduler.step(noise_pred_mv_flat, t, latents_mv_flat, generator=generator)
            latents_mv_prev = out_mv.prev_sample.view(1, num_frames, *out_mv.prev_sample.shape[1:])
            
            # Check if scheduler provided pred_original_sample (Diffusers > 0.26 usually does)
            if hasattr(out_2d, 'pred_original_sample') and out_2d.pred_original_sample is not None:
                x0_2d = out_2d.pred_original_sample
            else:
                # Fallback manual calculation for V-prediction: x0 = alpha * z - sigma * v
                # But Euler is simpler: x = x0 + sigma * n
                # With v-pred, it's safer to rely on the scheduler. 
                # If None, we skip coupling for this step to avoid exploding math.
                x0_2d = latents_2d # Fallback (no-op coupling)

            if hasattr(out_mv, 'pred_original_sample') and out_mv.pred_original_sample is not None:
                x0_mv = out_mv.pred_original_sample.view(1, num_frames, *out_mv.pred_original_sample.shape[1:])
            else:
                x0_mv = latents_mv # Fallback

            # --- Coupling Step ---
            # Scale coupling by sigma (noise level). High sigma = high noise = strong correction allowed.
            # Low sigma = fine details = weak correction.
            coupling_scale = sigma * lambda_

            x0_2d_resized = torch.nn.functional.interpolate(
                x0_2d, size=(72, 128), mode="bilinear", align_corners=False
            )
            x0_2d_expanded = x0_2d_resized.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

            x0_mv_mean = x0_mv.mean(dim=1, keepdim=True)
            x0_mv_mean_resized = torch.nn.functional.interpolate(
                 x0_mv_mean.squeeze(1), size=(64, 64), mode="bilinear", align_corners=False
            )

            # Apply Update
            latents_2d = latents_2d_prev - coupling_scale * (x0_2d - x0_mv_mean_resized)
            latents_mv = latents_mv_prev - coupling_scale * (x0_mv - x0_2d_expanded)

        return latents_2d, latents_mv

    def _encode_text(self, pipe, prompt, batch_size, device):
        tokens = pipe.tokenizer(
            prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        return pipe.text_encoder(tokens)[0].expand(batch_size, -1, -1)


def save_video(frames, path):
    import cv2
    height, width = np.array(frames[0]).shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 6, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    out.release()
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--prompt_2d", type=str, required=True)
    parser.add_argument("--ref_image_path", type=str, required=True)
    parser.add_argument("--model_2d_id", type=str, default="/fs/nexus-scratch/ltahboub/CoupledSceneSampling/mysore_palace_dreambooth_sd21")
    parser.add_argument("--model_mv_id", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--lambda_", type=float, default=0.1) # Adjusted for sigma scaling
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    # Load Models
    pipe_2d = StableDiffusionPipeline.from_pretrained(args.model_2d_id, torch_dtype=torch.float16)
    pipe_2d.to(device)

    pipe_mv = StableVideoDiffusionPipeline.from_pretrained(args.model_mv_id, torch_dtype=torch.float16)
    pipe_mv.to(device)

    sampler = CoupledDiffusion(pipe_2d, pipe_mv)
    ref_image = load_image(args.ref_image_path)

    print("Running coupled diffusion...")
    latents_2d, latents_mv = sampler(
        prompt_2d=args.prompt_2d,
        ref_image_mv=ref_image,
        lambda_=args.lambda_
    )

    with torch.no_grad():
        image_2d = pipe_2d.vae.decode(latents_2d / pipe_2d.vae.config.scaling_factor).sample
        image_2d = (image_2d / 2 + 0.5).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        Image.fromarray((image_2d * 255).astype("uint8")).save(f"{args.output_dir}/result_2d.png")

        latents_mv_flat = latents_mv.flatten(0, 1)
        frames_mv = pipe_mv.vae.decode(latents_mv_flat / pipe_mv.vae.config.scaling_factor, num_frames=latents_mv.shape[1]).sample
        frames_mv = (frames_mv / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
        frames_pil = [Image.fromarray((f * 255).astype("uint8")) for f in frames_mv]
        
        save_video(frames_pil, f"{args.output_dir}/result_mv.mp4")

    print(f"Saved to {args.output_dir}")

if __name__ == "__main__":
    main()