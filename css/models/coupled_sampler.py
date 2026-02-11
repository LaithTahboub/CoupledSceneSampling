"""
Coupled diffusion sampling: SEVA + Pose-Conditioned SD.
SEVA provides geometry, SD provides prompt-following + reference appearance.
Coupling harmonizes x0 predictions.
"""

import torch
import numpy as np
from PIL import Image

from css.models.pose_conditioned_sd import PoseConditionedSD

from seva.utils import load_model as seva_load_model
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DDPMDiscretization, DiscreteDenoiser, EulerEDMSampler, VanillaCFG
from seva.geometry import (
    get_plucker_coordinates, to_hom_pose, get_default_intrinsics,
    get_preset_pose_fov, DEFAULT_FOV_RAD
)


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    return x[(...,) + (None,) * (target_dims - x.ndim)]


class CoupledDiffusionSampler:

    def __init__(
        self,
        dreambooth_path: str,
        pose_sd_checkpoint: str | None = None,
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

        # SEVA
        self.seva_model = SGMWrapper(seva_load_model(device="cpu").eval()).to(device)
        self.ae = AutoEncoder(chunk_size=1).to(device)
        self.clip_cond = CLIPConditioner().to(device)
        self.discretization = DDPMDiscretization()
        self.denoiser = DiscreteDenoiser(num_idx=1000, device=device)

        # Pose-conditioned SD with cross-frame attention
        self.pose_sd = PoseConditionedSD(pretrained_model=dreambooth_path, device=device)
        if pose_sd_checkpoint:
            state = torch.load(pose_sd_checkpoint, map_location=device)
            if isinstance(state, dict) and "unet" in state:
                self.pose_sd.unet.load_state_dict(state["unet"])
                self.pose_sd.ref_encoder.load_state_dict(state["ref_encoder"])
            else:
                # Old format: just UNet state_dict
                self.pose_sd.unet.load_state_dict(state)
            print(f"Loaded pose-SD checkpoint: {pose_sd_checkpoint}")
        self.pose_sd.eval()
        self.sd_alphas = self.pose_sd.scheduler.alphas_cumprod.to(device)

    def _parse_images(self, images):
        """Normalize input to (image_list, index_list)."""
        if isinstance(images, Image.Image):
            images = [(images, 0)]
        return [img for img, _ in images], [idx for _, idx in images]

    def _preprocess_image(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("RGB").resize((self.W, self.H))) / 255.0
        return (torch.from_numpy(arr).permute(2, 0, 1).float() * 2 - 1).unsqueeze(0).to(self.device)

    def _encode_images(self, images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images for SEVA (SEVA's VAE + CLIP)."""
        latents, clip_embs = [], []
        for img in images:
            tensor = self._preprocess_image(img)
            with torch.autocast("cuda"):
                latents.append(self.ae.encode(tensor, 1))
                clip_embs.append(self.clip_cond(tensor).mean(0))
        return torch.cat(latents, dim=0), torch.stack(clip_embs)

    def _encode_sd_refs(self, images: list[Image.Image]) -> torch.Tensor:
        """Encode input images with SD's VAE for reference conditioning."""
        latents = []
        for img in images:
            tensor = self._preprocess_image(img)
            with torch.no_grad():
                latents.append(self.pose_sd.encode_image(tensor))
        return torch.cat(latents, dim=0)  # (num_inputs, 4, h, w)

    def _build_trajectory(self, num_frames: int, trajectory: str):
        """Returns (w2c, pluckers, all_Ks)."""
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
        w2c[:, :3, 3] *= 2.0

        pluckers = get_plucker_coordinates(
            extrinsics_src=w2c[0], extrinsics=w2c,
            intrinsics=all_Ks, target_size=[self.latent_h, self.latent_w]
        )
        return w2c, pluckers, all_Ks

    def _build_conditioning(self, num_frames, input_indices, encoded_latents, clip_embeddings, pluckers):
        """Build SEVA conditioning dicts."""
        N = num_frames
        h, w = self.latent_h, self.latent_w

        c_crossattn = torch.zeros(N, 1, clip_embeddings.shape[-1], device=self.device)
        for j in range(N):
            nearest = min(input_indices, key=lambda idx: abs(idx - j))
            c_crossattn[j] = clip_embeddings[input_indices.index(nearest)].unsqueeze(0)

        input_mask = torch.zeros(N, 1, h, w, device=self.device)
        c_replace = torch.zeros(N, 5, h, w, device=self.device)
        for i, idx in enumerate(input_indices):
            input_mask[idx] = 1.0
            c_replace[idx, :4] = encoded_latents[i]
            c_replace[idx, 4:] = 1.0

        c_concat = torch.cat([input_mask, pluckers], dim=1)
        uc_concat = torch.cat([torch.zeros_like(input_mask), pluckers], dim=1)

        c = {"crossattn": c_crossattn, "replace": c_replace, "concat": c_concat, "dense_vector": pluckers}
        uc = {"crossattn": torch.zeros_like(c_crossattn), "replace": torch.zeros_like(c_replace),
              "concat": uc_concat, "dense_vector": pluckers}
        return c, uc

    def _seva_x0(self, x, sigma, c, uc, N):
        x2 = torch.cat([x, x])
        s2 = torch.cat([sigma, sigma])
        cond = {k: torch.cat([uc[k], c[k]]) for k in c}
        with torch.autocast("cuda"):
            pred = self.denoiser(self.seva_model, x2, s2, cond, num_frames=N)
        uc_pred, c_pred = pred.chunk(2)
        return uc_pred + self.cfg_seva * (c_pred - uc_pred)

    def _sd_x0(self, x, t, prompt_emb, w2c, all_Ks, input_indices, sd_ref_latents):
        """Predict x0 from SD with cross-frame attention to closest references.

        Args:
            x: (N, 4, h, w) noisy latents for all frames
            t: SD timestep (int)
            prompt_emb: (2, 77, dim) - [prompt_cond, null_text]
            w2c: (N, 4, 4) world-to-camera for all trajectory frames
            all_Ks: (N, 3, 3) intrinsics for all trajectory frames
            input_indices: list of frame indices that have input images
            sd_ref_latents: (num_inputs, 4, h, w) SD-encoded input image latents
        """
        alpha_bar = self.sd_alphas[t]
        h, w = self.latent_h, self.latent_w
        x0_list = []

        for j in range(x.shape[0]):
            # Find 2 closest input frames by trajectory index
            closest = sorted(input_indices, key=lambda idx: abs(idx - j))[:2]
            if len(closest) == 1:
                closest = [closest[0], closest[0]]

            ref1_frame_idx, ref2_frame_idx = closest[0], closest[1]
            ref1_latent = sd_ref_latents[input_indices.index(ref1_frame_idx)].unsqueeze(0)
            ref2_latent = sd_ref_latents[input_indices.index(ref2_frame_idx)].unsqueeze(0)

            # Match training convention: express all Pluckers in ref1 frame.
            src_w2c = w2c[ref1_frame_idx]
            plucker_ref1 = get_plucker_coordinates(
                extrinsics_src=src_w2c, extrinsics=w2c[ref1_frame_idx].unsqueeze(0),
                intrinsics=all_Ks[ref1_frame_idx].unsqueeze(0), target_size=[h, w]
            )
            plucker_ref2 = get_plucker_coordinates(
                extrinsics_src=src_w2c, extrinsics=w2c[ref2_frame_idx].unsqueeze(0),
                intrinsics=all_Ks[ref2_frame_idx].unsqueeze(0), target_size=[h, w]
            )
            plucker_target = get_plucker_coordinates(
                extrinsics_src=src_w2c, extrinsics=w2c[j].unsqueeze(0),
                intrinsics=all_Ks[j].unsqueeze(0), target_size=[h, w]
            )

            # Encode refs to cross-attention tokens (separate call per ref)
            ref1_tokens = self.pose_sd.ref_encoder(ref1_latent, plucker_ref1)  # (1, 256, 1024)
            ref2_tokens = self.pose_sd.ref_encoder(ref2_latent, plucker_ref2)  # (1, 256, 1024)
            ref_tokens = torch.cat([ref1_tokens, ref2_tokens], dim=1)  # (1, 512, 1024)

            # Build conditional and unconditional cross-attention inputs
            cond = torch.cat([prompt_emb[0:1], ref_tokens], dim=1)
            uncond = torch.cat([prompt_emb[1:2], self.pose_sd.null_ref_tokens], dim=1)

            # 10-channel UNet input: noisy latent + target Plucker
            unet_input = torch.cat([x[j:j+1], plucker_target], dim=1)  # (1, 10, h, w)

            # CFG: run both unconditional and conditional
            x_in = torch.cat([unet_input, unet_input])
            t_in = torch.tensor([t, t], device=self.device)
            enc_hidden = torch.cat([uncond, cond])

            with torch.autocast("cuda"):
                eps = self.pose_sd.unet(
                    x_in.half(), t_in, encoder_hidden_states=enc_hidden.half()
                ).sample.float()

            eps_uc, eps_c = eps.chunk(2)
            eps_cfg = eps_uc + self.cfg_sd * (eps_c - eps_uc)
            x0_list.append((x[j:j+1] - (1 - alpha_bar).sqrt() * eps_cfg) / alpha_bar.sqrt())

        return torch.cat(x0_list)

    def _sigma_to_sd_t(self, sigma):
        if sigma <= 0:
            return 0
        sigma_sd = sigma / np.exp(2.4)
        alpha_bar = 1.0 / (1.0 + sigma_sd.item() ** 2)
        return (self.sd_alphas - alpha_bar).abs().argmin().item()

    def _decode_frames(self, latents):
        results = []
        for j in range(latents.shape[0]):
            with torch.autocast("cuda"):
                decoded = self.ae.decode(latents[j:j+1], 1)
            out = ((decoded.clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(1, 2, 0).cpu().numpy()
            results.append(Image.fromarray(out))
        return results

    @torch.inference_mode()
    def sample_seva_only(self, images, trajectory="orbit", num_frames=21, seed=42):
        """SEVA-only sampling (no SD coupling)."""
        torch.manual_seed(seed)
        input_images, input_indices = self._parse_images(images)

        encoded_latents, clip_embeddings = self._encode_images(input_images)
        _, pluckers, _ = self._build_trajectory(num_frames, trajectory)
        c, uc = self._build_conditioning(num_frames, input_indices, encoded_latents, clip_embeddings, pluckers)

        sampler = EulerEDMSampler(
            discretization=self.discretization, guider=VanillaCFG(),
            num_steps=self.num_steps, verbose=True, device=self.device,
        )
        randn = torch.randn(num_frames, 4, self.latent_h, self.latent_w, device=self.device)

        with torch.autocast("cuda"):
            samples_z = sampler(
                lambda inp, sigma, cond: self.denoiser(self.seva_model, inp, sigma, cond, num_frames=num_frames),
                randn, scale=self.cfg_seva, cond=c, uc=uc,
            )
        return self._decode_frames(samples_z)

    @torch.inference_mode()
    def sample(self, images, prompt, trajectory="orbit", num_frames=21, seed=42):
        """Coupled SEVA + Pose-Conditioned SD sampling."""
        torch.manual_seed(seed)
        input_images, input_indices = self._parse_images(images)

        # SEVA encoding
        encoded_latents, clip_embeddings = self._encode_images(input_images)
        w2c, pluckers, all_Ks = self._build_trajectory(num_frames, trajectory)
        c, uc = self._build_conditioning(num_frames, input_indices, encoded_latents, clip_embeddings, pluckers)

        # SD reference encoding (input images -> SD VAE latents)
        sd_ref_latents = self._encode_sd_refs(input_images)

        # SD text embeddings: [prompt_cond, null_text]
        text_cond = self.pose_sd.get_text_embedding(prompt)
        text_uncond = self.pose_sd.null_text_emb
        prompt_emb = torch.cat([text_cond, text_uncond], dim=0)  # (2, 77, dim)

        sigmas = self.discretization(self.num_steps, device=self.device)
        noise = torch.randn(num_frames, 4, self.latent_h, self.latent_w, device=self.device)
        x_seva = noise * torch.sqrt(1.0 + sigmas[0] ** 2)
        x_sd = noise.clone()

        for i in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            sd_t = self._sigma_to_sd_t(sigma)
            sd_t_next = self._sigma_to_sd_t(sigma_next)

            sigma_batch = sigma.expand(num_frames)
            x0_seva = self._seva_x0(x_seva, sigma_batch, c, uc, num_frames)
            x0_sd = self._sd_x0(x_sd, sd_t, prompt_emb, w2c, all_Ks, input_indices, sd_ref_latents)

            grad = self.coupling_strength * (x0_seva - x0_sd)
            x0_seva_c = x0_seva - grad
            x0_sd_c = x0_sd + grad

            d = (x_seva - x0_seva_c) / append_dims(sigma, x_seva.ndim)
            x_seva = x_seva + (sigma_next - sigma) * d

            if sd_t_next > 0:
                alpha_next = self.sd_alphas[sd_t_next]
                x_sd = alpha_next.sqrt() * x0_sd_c + (1 - alpha_next).sqrt() * torch.randn_like(x_sd)
            else:
                x_sd = x0_sd_c

        return self._decode_frames(x_seva)


if __name__ == "__main__":
    import argparse
    import imageio

    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="'path:frame_idx' pairs")
    p.add_argument("--dreambooth", default="manojb/stable-diffusion-2-1-base")
    p.add_argument("--pose-sd-checkpoint", default=None, help="Path to trained UNet checkpoint (.pt)")
    p.add_argument("--prompt", default="")
    p.add_argument("--trajectory", default="orbit", choices=["orbit", "spiral"])
    p.add_argument("--frames", type=int, default=21)
    p.add_argument("--output", default="output.mp4")
    p.add_argument("--coupling", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare-seva", action="store_true")
    args = p.parse_args()

    images = []
    for inp in args.inputs:
        if ":" in inp:
            path, idx = inp.rsplit(":", 1)
            images.append((Image.open(path), int(idx)))
        else:
            images.append((Image.open(inp), 0))

    sampler = CoupledDiffusionSampler(
        args.dreambooth,
        pose_sd_checkpoint=args.pose_sd_checkpoint,
        coupling_strength=args.coupling,
    )

    results = sampler.sample(images, args.prompt, args.trajectory, args.frames, args.seed)
    imageio.mimsave(args.output, [np.array(img) for img in results], fps=30)
    print(f"Saved coupled: {args.output}")

    if args.compare_seva:
        seva_output = args.output.replace(".mp4", "_seva_only.mp4")
        seva_results = sampler.sample_seva_only(images, args.trajectory, args.frames, args.seed)
        imageio.mimsave(seva_output, [np.array(img) for img in seva_results], fps=30)
        print(f"Saved SEVA-only: {seva_output}")
