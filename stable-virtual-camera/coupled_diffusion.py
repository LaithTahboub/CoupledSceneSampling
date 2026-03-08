"""
Coupled Diffusion Sampling for SEVA + DreamBooth SD2.1

Supports multiple input images from different views/conditions.
SEVA provides geometric consistency, SD provides prompt-following.
Coupling harmonizes appearance across all views.
"""

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline

from seva.utils import load_model as seva_load_model
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DDPMDiscretization, DiscreteDenoiser
from seva.geometry import (
    get_plucker_coordinates, to_hom_pose, get_default_intrinsics,
    get_preset_pose_fov, DEFAULT_FOV_RAD
)


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    return x[(...,) + (None,) * (target_dims - x.ndim)]


class CoupledDiffusionSampler:
    """
    Couples SEVA (multi-view) with DreamBooth SD2.1 (2D) through x0 predictions.

    Multi-image workflow:
    1. Input images anchor SEVA at specified frame positions
    2. SD generates all frames according to prompt
    3. Coupling pulls x0 predictions together, harmonizing appearance
    """

    def __init__(
        self,
        dreambooth_path: str,
        coupling_strength: float = 0.5,
        cfg_seva: float = 2.0,
        cfg_sd: float = 7.5,
        num_steps: int = 50,
        H: int = 576,
        W: int = 576,
        device: str = "cuda",
    ):
        self.coupling_strength = coupling_strength
        self.cfg_seva = cfg_seva
        self.cfg_sd = cfg_sd
        self.num_steps = num_steps
        self.H, self.W = H, W
        self.device = device
        self.latent_h, self.latent_w = H // 8, W // 8

        # SEVA components
        self.seva_model = SGMWrapper(seva_load_model(device="cpu").eval()).to(device)
        self.ae = AutoEncoder(chunk_size=1).to(device)
        self.clip_cond = CLIPConditioner().to(device)
        self.discretization = DDPMDiscretization()
        self.denoiser = DiscreteDenoiser(num_idx=1000, device=device)

        # DreamBooth SD2.1
        self.sd = StableDiffusionPipeline.from_pretrained(
            dreambooth_path, torch_dtype=torch.float16, safety_checker=None
        ).to(device)
        self.sd_alphas = self.sd.scheduler.alphas_cumprod.to(device)

    def _preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor."""
        arr = np.array(img.convert("RGB").resize((self.W, self.H))) / 255.0
        return (torch.from_numpy(arr).permute(2, 0, 1).float() * 2 - 1).unsqueeze(0).to(self.device)

    def _encode_images(self, images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images to latents and CLIP embeddings."""
        latents, clip_embs = [], []
        for img in images:
            tensor = self._preprocess_image(img)
            with torch.autocast("cuda"):
                latents.append(self.ae.encode(tensor, 1))
                clip_embs.append(self.clip_cond(tensor).mean(0))
        return torch.cat(latents, dim=0), torch.stack(clip_embs)

    def _build_trajectory(self, num_frames: int, trajectory: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Build camera trajectory and Plucker coordinates."""
        c2w_start = torch.eye(4)
        all_c2ws, all_fovs = get_preset_pose_fov(
            trajectory, num_frames, c2w_start, torch.tensor([0, 0, 10.0]),
            -c2w_start[:3, 1], DEFAULT_FOV_RAD, spiral_radii=[1.0, 1.0, 0.5], zoom_factor=None
        )
        all_c2ws = torch.as_tensor(all_c2ws, device=self.device, dtype=torch.float32)
        all_fovs = torch.as_tensor(all_fovs, device=self.device)
        all_Ks = get_default_intrinsics(all_fovs, self.W / self.H).to(self.device)

        c2w = to_hom_pose(all_c2ws)
        c2w[:, :3, 3] -= c2w[:, :3, 3].mean(0, keepdim=True)
        w2c = torch.linalg.inv(c2w)
        w2c[:, :3, 3] *= 2.0  # camera_scale

        pluckers = get_plucker_coordinates(
            extrinsics_src=w2c[0], extrinsics=w2c,
            intrinsics=all_Ks, target_size=[self.latent_h, self.latent_w]
        )
        return w2c, pluckers

    def _build_conditioning(
        self,
        num_frames: int,
        input_indices: list[int],
        encoded_latents: torch.Tensor,
        clip_embeddings: torch.Tensor,
        pluckers: torch.Tensor,
    ) -> tuple[dict, dict]:
        """Build SEVA conditioning dicts for input images at specified positions."""
        N = num_frames
        h, w = self.latent_h, self.latent_w

        # CLIP: use nearest input image's embedding for each frame
        c_crossattn = torch.zeros(N, 1, clip_embeddings.shape[-1], device=self.device)
        for j in range(N):
            nearest = min(input_indices, key=lambda idx: abs(idx - j))
            c_crossattn[j] = clip_embeddings[input_indices.index(nearest)].unsqueeze(0)

        # Replace: anchor input images at their positions
        input_mask = torch.zeros(N, 1, h, w, device=self.device)
        c_replace = torch.zeros(N, 5, h, w, device=self.device)

        for i, idx in enumerate(input_indices):
            input_mask[idx] = 1.0
            c_replace[idx, :4] = encoded_latents[i]
            c_replace[idx, 4:] = 1.0  # mask channel

        c_concat = torch.cat([input_mask, pluckers], dim=1)
        uc_concat = torch.cat([torch.zeros_like(input_mask), pluckers], dim=1)

        c = {"crossattn": c_crossattn, "replace": c_replace, "concat": c_concat, "dense_vector": pluckers}
        uc = {"crossattn": torch.zeros_like(c_crossattn), "replace": torch.zeros_like(c_replace),
              "concat": uc_concat, "dense_vector": pluckers}
        return c, uc

    def _seva_x0(self, x: torch.Tensor, sigma: torch.Tensor, c: dict, uc: dict, N: int) -> torch.Tensor:
        """SEVA x0 prediction with CFG."""
        x2 = torch.cat([x, x])
        s2 = torch.cat([sigma, sigma])
        cond = {k: torch.cat([uc[k], c[k]]) for k in c}

        with torch.autocast("cuda"):
            pred = self.denoiser(self.seva_model, x2, s2, cond, num_frames=N)

        uc_pred, c_pred = pred.chunk(2)
        return uc_pred + self.cfg_seva * (c_pred - uc_pred)

    def _sd_x0(self, x: torch.Tensor, t: int, prompt_emb: torch.Tensor) -> torch.Tensor:
        """SD x0 prediction with CFG for all frames."""
        alpha_bar = self.sd_alphas[t]
        x0_list = []

        for j in range(x.shape[0]):
            x_in = torch.cat([x[j:j+1], x[j:j+1]])
            t_in = torch.tensor([t, t], device=self.device)

            with torch.autocast("cuda"):
                eps = self.sd.unet(x_in.half(), t_in, encoder_hidden_states=prompt_emb.half()).sample.float()

            eps_uc, eps_c = eps.chunk(2)
            eps_cfg = eps_uc + self.cfg_sd * (eps_c - eps_uc)
            x0 = (x[j:j+1] - (1 - alpha_bar).sqrt() * eps_cfg) / alpha_bar.sqrt()
            x0_list.append(x0)

        return torch.cat(x0_list)

    def _sigma_to_sd_t(self, sigma: torch.Tensor) -> int:
        """Convert SEVA sigma to SD timestep."""
        if sigma <= 0:
            return 0
        sigma_sd = sigma / np.exp(2.4)  # Remove log_snr_shift
        alpha_bar = 1.0 / (1.0 + sigma_sd.item() ** 2)
        return (self.sd_alphas - alpha_bar).abs().argmin().item()

    @torch.inference_mode()
    def sample(
        self,
        images: Image.Image | list[tuple[Image.Image, int]],
        prompt: str,
        trajectory: str = "orbit",
        num_frames: int = 21,
        seed: int = 42,
    ) -> list[Image.Image]:
        """
        Coupled diffusion sampling with single or multiple input images.

        Args:
            images: Single image (placed at frame 0) or list of (image, frame_index) tuples
            prompt: Text prompt for SD (e.g., "clear sky, no tourists")
            trajectory: "orbit" or "spiral"
            num_frames: Total frames to generate
            seed: Random seed

        Returns:
            List of generated PIL images
        """
        torch.manual_seed(seed)

        # Normalize input
        if isinstance(images, Image.Image):
            image_list = [(images, 0)]
        else:
            image_list = images

        input_images = [img for img, _ in image_list]
        input_indices = [idx for _, idx in image_list]

        print(f"Input images at frames: {input_indices}")

        # Encode inputs
        encoded_latents, clip_embeddings = self._encode_images(input_images)

        # Build trajectory
        _, pluckers = self._build_trajectory(num_frames, trajectory)

        # Build conditioning
        c, uc = self._build_conditioning(
            num_frames, input_indices, encoded_latents, clip_embeddings, pluckers
        )

        # SD prompt embeddings
        tok = self.sd.tokenizer([prompt, ""], padding="max_length", max_length=77, return_tensors="pt")
        prompt_emb = self.sd.text_encoder(tok.input_ids.to(self.device))[0]

        # Initialize
        sigmas = self.discretization(self.num_steps, device=self.device)
        noise = torch.randn(num_frames, 4, self.latent_h, self.latent_w, device=self.device)

        x_seva = noise * torch.sqrt(1.0 + sigmas[0] ** 2)
        x_sd = noise.clone()

        print(f"Coupled sampling: {self.num_steps} steps, {num_frames} frames, λ={self.coupling_strength}")

        # Sampling loop
        for i in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            sd_t = self._sigma_to_sd_t(sigma)
            sd_t_next = self._sigma_to_sd_t(sigma_next)

            # Get x0 predictions
            sigma_batch = sigma.expand(num_frames)
            x0_seva = self._seva_x0(x_seva, sigma_batch, c, uc, num_frames)
            x0_sd = self._sd_x0(x_sd, sd_t, prompt_emb)

            # Coupling: pull x0 predictions toward each other
            grad = self.coupling_strength * (x0_seva - x0_sd)
            x0_seva_c = x0_seva - grad
            x0_sd_c = x0_sd + grad

            # SEVA Euler step
            d = (x_seva - x0_seva_c) / append_dims(sigma, x_seva.ndim)
            x_seva = x_seva + (sigma_next - sigma) * d

            # SD DDPM step
            if sd_t_next > 0:
                alpha_next = self.sd_alphas[sd_t_next]
                x_sd = alpha_next.sqrt() * x0_sd_c + (1 - alpha_next).sqrt() * torch.randn_like(x_sd)
            else:
                x_sd = x0_sd_c

            if (i + 1) % 10 == 0:
                print(f"  Step {i+1}/{self.num_steps}")

        # Decode
        results = []
        for j in range(num_frames):
            with torch.autocast("cuda"):
                decoded = self.ae.decode(x_seva[j:j+1], 1)
            out = ((decoded.clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(1, 2, 0).cpu().numpy()
            results.append(Image.fromarray(out))

        return results


if __name__ == "__main__":
    import argparse
    import imageio
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Input images as 'path:frame_idx' (e.g., 'img1.jpg:0 img2.jpg:10')")
    p.add_argument("--dreambooth", default="/fs/nexus-scratch/ltahboub/CoupledSceneSampling/mysore_palace_dreambooth_sd21", help="Path to DreamBooth model")
    p.add_argument("--prompt", default="a photo of sks palace, clear sky, no tourists", help="Text prompt")
    p.add_argument("--trajectory", default="orbit", choices=["orbit", "spiral"])
    p.add_argument("--frames", type=int, default=21)
    p.add_argument("--output", default="output.mp4")
    p.add_argument("--coupling", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Parse inputs
    images = []
    for inp in args.inputs:
        if ":" in inp:
            path, idx = inp.rsplit(":", 1)
            images.append((Image.open(path), int(idx)))
        else:
            images.append((Image.open(inp), 0))

    sampler = CoupledDiffusionSampler(args.dreambooth, coupling_strength=args.coupling)
    results = sampler.sample(images, args.prompt, args.trajectory, args.frames, args.seed)

    imageio.mimsave(args.output, [np.array(img) for img in results], fps=30)
    print(f"Saved {args.output}")
