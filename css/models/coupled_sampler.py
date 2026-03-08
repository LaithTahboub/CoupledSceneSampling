"""
Coupled diffusion sampling: SEVA + PoseSD.
SEVA provides geometry, PoseSD provides prompt-following + reference appearance.
Coupling harmonizes x0 predictions at each denoising step.
"""

import copy
import os
import tempfile
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange, repeat

from css.models.pose_sd import PoseSD
from css.models.EMA import load_pose_sd_checkpoint

from seva.utils import load_model as seva_load_model
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.modules.preprocessor import Dust3rPipeline
from seva.sampling import (
    DDPMDiscretization,
    DiscreteDenoiser,
    MultiviewCFG,
    append_dims,
    to_d,
)
from seva.eval import (
    run_one_scene,
    infer_prior_stats,
    transform_img_and_K,
)
from seva.geometry import (
    generate_interpolated_path,
    get_plucker_coordinates,
    to_hom_pose,
    get_default_intrinsics,
    get_preset_pose_fov,
    normalize_scene,
    DEFAULT_FOV_RAD,
)


VERSION_DICT = {
    "H": 576,
    "W": 576,
    "T": 21,
    "C": 4,
    "f": 8,
    "options": {},
}


class CoupledDiffusionSampler:

    def __init__(
        self,
        pretrained_path: str,
        pose_sd_checkpoint: str | None = None,
        coupling_strength: float = 0.5,
        cfg_seva: float = 3.0,
        cfg_sd: float = 7.5,
        cfg_min: float = 1.2,
        camera_scale: float = 2.0,
        num_steps: int = 50,
        chunk_strategy: str = "interp-gt",
        H: int = 576,
        W: int = 576,
        device: str = "cuda",
    ):
        self.coupling_strength = coupling_strength
        self.num_steps = num_steps
        self.H, self.W = H, W
        self.device = device
        self.latent_h, self.latent_w = H // 8, W // 8
        self.cfg_seva = float(cfg_seva)
        self.cfg_sd = float(cfg_sd)
        self.cfg_min = cfg_min
        self.camera_scale = camera_scale
        self.chunk_strategy = chunk_strategy

        # SEVA
        self.seva_model = SGMWrapper(seva_load_model(device="cpu").eval()).to(device)
        self.ae = AutoEncoder(chunk_size=1).to(device)
        self.clip_cond = CLIPConditioner().to(device)
        self.discretization = DDPMDiscretization()
        self.denoiser = DiscreteDenoiser(num_idx=1000, device=device)

        # DUSt3R for pose estimation
        self.dust3r = Dust3rPipeline(device=device)

        # PoseSD with cross-view attention
        self.pose_sd = PoseSD(pretrained_model=pretrained_path, device=device)
        if pose_sd_checkpoint:
            load_pose_sd_checkpoint(self.pose_sd, pose_sd_checkpoint, device)
            print(f"Loaded PoseSD checkpoint: {pose_sd_checkpoint}")
        self.pose_sd.eval()
        self.sd_alphas = self.pose_sd.scheduler.alphas_cumprod.to(device)

    @staticmethod
    def _x0_from_eps(latent, eps, alpha_bar):
        sqrt_alpha = alpha_bar.sqrt()
        sqrt_one_minus = (1.0 - alpha_bar).clamp_min(0.0).sqrt()
        return (latent - sqrt_one_minus * eps) / sqrt_alpha

    @staticmethod
    def _cfg_combine(cond, uncond, guidance_scale):
        if guidance_scale <= 1.0:
            return cond
        return uncond + guidance_scale * (cond - uncond)

    def _autocast_context(self):
        return torch.autocast("cuda") if str(self.device).startswith("cuda") else nullcontext()

    def _parse_images(self, images):
        """Normalize input to (path_or_pil_list, index_list).

        Accepts:
          - Single PIL image
          - List of (PIL_or_path, frame_idx) tuples
        """
        if isinstance(images, (Image.Image, str)):
            images = [(images, 0)]
        return [img for img, _ in images], [idx for _, idx in images]

    def _images_to_paths(self, images):
        """Ensure all images are saved as files (needed for DUSt3R).

        Always saves to temp PNGs — DUSt3R's load_images filters by extension
        and needs files it can open directly.
        """
        paths = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.convert("RGB").save(f.name)
            paths.append(f.name)
        return paths

    def _preprocess_inputs_for_seva(self, img_paths):
        """Replicate demo_gr.py Advanced preprocess exactly."""
        shorter = round(576 / 64) * 64
        input_imgs, input_Ks, input_c2ws, points, _ = self.dust3r.infer_cameras_and_points(
            img_paths
        )
        num_inputs = len(img_paths)
        if num_inputs == 1:
            input_imgs, input_Ks, input_c2ws, points = (
                input_imgs[:1],
                input_Ks[:1],
                input_c2ws[:1],
                points[:1],
            )

        input_imgs = [img[..., :3] for img in input_imgs]

        point_chunks = [p.shape[0] for p in points]
        point_indices = np.cumsum(point_chunks)[:-1]
        input_c2ws, points, _ = normalize_scene(
            input_c2ws,
            np.concatenate(points, 0),
            camera_center_method="poses",
        )
        points = np.split(points, point_indices, 0)

        scene_scale = np.median(
            np.ptp(np.concatenate([input_c2ws[:, :3, 3], *points], 0), -1)
        )
        input_c2ws[:, :3, 3] /= scene_scale

        input_imgs = [torch.as_tensor(img / 255.0, dtype=torch.float32) for img in input_imgs]
        input_Ks = torch.as_tensor(input_Ks, dtype=torch.float32)
        input_c2ws = torch.as_tensor(input_c2ws, dtype=torch.float32)

        resized_imgs, resized_Ks = [], []
        for img, K in zip(input_imgs, input_Ks):
            img = rearrange(img, "h w c -> 1 c h w")
            img, K = transform_img_and_K(img, shorter, K=K[None], size_stride=64)
            assert isinstance(K, torch.Tensor)
            K = K / K.new_tensor([img.shape[-1], img.shape[-2], 1])[:, None]
            resized_imgs.append(img)
            resized_Ks.append(K)

        input_imgs = torch.cat(resized_imgs, 0)
        input_imgs = rearrange(input_imgs, "b c h w -> b h w c")[..., :3]
        input_Ks = torch.cat(resized_Ks, 0)
        input_wh = (input_imgs.shape[2], input_imgs.shape[1])  # (W, H)
        return input_imgs, input_Ks, input_c2ws, input_wh

    def _build_unified_trajectory(self, input_c2ws, input_Ks, num_frames, input_wh):
        """Build a single trajectory of num_frames with inputs at keyframe positions.

        For N inputs and F total frames, input images are placed at evenly-spaced
        positions [0, F/(N-1), 2*F/(N-1), ..., F-1]. The gaps are filled with
        interpolated poses.

        Returns (all_c2ws, all_Ks, input_indices) — all in temporal order.
        """
        n_inputs = input_c2ws.shape[0]
        if n_inputs < 2:
            # Single input: orbit around it
            c2w_start = input_c2ws[0]
            target_c2ws, target_fovs = get_preset_pose_fov(
                "orbit", num_frames,
                torch.linalg.inv(c2w_start),
                torch.tensor([0, 0, 10.0]),
                -c2w_start[:3, 1],
                DEFAULT_FOV_RAD,
                spiral_radii=[0.3, 0.3, 0.1],
                zoom_factor=None,
            )
            all_c2ws = torch.as_tensor(target_c2ws, dtype=torch.float32)
            all_Ks = get_default_intrinsics(
                torch.as_tensor(target_fovs), input_wh[0] / input_wh[1]
            )
            # Input at first frame
            all_c2ws[0] = to_hom_pose(input_c2ws[:1])[0]
            all_Ks[0] = input_Ks[0]
            return all_c2ws, all_Ks, [0]

        # Interpolate between input poses to get num_frames total
        keyframe_c2ws = input_c2ws.numpy()
        n_interp = max(2, -(-num_frames // (n_inputs - 1)))
        interp_c2ws = generate_interpolated_path(
            keyframe_c2ws[:, :3],
            n_interp=n_interp,
            spline_degree=min(5, n_inputs - 1),
            smoothness=0,
            endpoint=True,
        )[:num_frames]

        all_c2ws = torch.as_tensor(interp_c2ws, dtype=torch.float32)
        all_c2ws = to_hom_pose(all_c2ws)
        all_Ks = input_Ks[:1].expand(num_frames, -1, -1).clone()

        # Input keyframe positions: evenly spaced
        input_indices = [
            round(i * (num_frames - 1) / (n_inputs - 1)) for i in range(n_inputs)
        ]

        # Snap keyframe positions to exact input poses/Ks
        for i, idx in enumerate(input_indices):
            all_c2ws[idx] = to_hom_pose(input_c2ws[i:i+1])[0]
            all_Ks[idx] = input_Ks[i]

        return all_c2ws, all_Ks, input_indices

    def _build_target_trajectory(self, input_c2ws, input_Ks, num_targets):
        num_inputs = input_c2ws.shape[0]
        if num_targets <= 0:
            return input_c2ws.new_zeros((0, 4, 4)), input_Ks.new_zeros((0, 3, 3))

        if num_inputs >= 2:
            keyframe_c2ws = input_c2ws.numpy()
            n_interp = max(2, int(np.ceil(num_targets / (num_inputs - 1))))
            target_c2ws_np = generate_interpolated_path(
                keyframe_c2ws[:, :3],
                n_interp=n_interp,
                spline_degree=min(5, num_inputs - 1),
                smoothness=0,
                endpoint=True,
            )
            sample_idx = (
                np.linspace(0, len(target_c2ws_np) - 1, num_targets).round().astype(int)
            )
            target_c2ws = to_hom_pose(
                torch.as_tensor(target_c2ws_np[sample_idx], dtype=torch.float32)
            )
            target_Ks = input_Ks[:1].expand(num_targets, -1, -1).clone()
            return target_c2ws, target_Ks

        orbit_c2ws, _ = get_preset_pose_fov(
            "orbit",
            num_targets + 1,
            torch.linalg.inv(input_c2ws[0]),
            torch.tensor([0, 0, 10.0]),
            -input_c2ws[0, :3, 1],
            DEFAULT_FOV_RAD,
            spiral_radii=[0.3, 0.3, 0.1],
            zoom_factor=None,
        )
        target_c2ws = to_hom_pose(
            torch.as_tensor(orbit_c2ws[1 : num_targets + 1], dtype=torch.float32)
        )
        target_Ks = input_Ks[:1].expand(num_targets, -1, -1).clone()
        return target_c2ws, target_Ks

    def _build_run_one_scene_args(self, img_paths, num_frames):
        """Build run_one_scene args aligned with demo_gr.py's Advanced render path."""
        input_imgs, input_Ks, input_c2ws, (W, H) = self._preprocess_inputs_for_seva(
            img_paths
        )
        num_inputs = len(img_paths)
        num_total_frames = max(int(num_frames), num_inputs + 1)
        num_targets = num_total_frames - num_inputs
        target_c2ws, target_Ks = self._build_target_trajectory(
            input_c2ws, input_Ks, num_targets
        )

        # Set up options early (infer_prior_stats reads chunk_strategy from it)
        options = {
            "chunk_strategy": self.chunk_strategy,
            "video_save_fps": 30.0,
            "beta_linear_start": 5e-6,
            "log_snr_shift": 2.4,
            "guider_types": [1, 2],
            "cfg": [self.cfg_seva, 3.0 if num_inputs >= 9 else 2.0],
            "camera_scale": self.camera_scale,
            "num_steps": self.num_steps,
            "cfg_min": self.cfg_min,
            "encoding_t": 1,
            "decoding_t": 1,
        }

        T_base = VERSION_DICT["T"]
        version_dict = copy.deepcopy(VERSION_DICT)
        version_dict["H"] = H
        version_dict["W"] = W
        version_dict["options"] = options
        all_c2ws = torch.cat([input_c2ws, target_c2ws], 0)
        all_Ks = torch.cat([input_Ks, target_Ks], 0)
        all_Ks_pixel = all_Ks * all_Ks.new_tensor([W, H, 1.0])[:, None]

        # input_indices: [0, 1, ..., num_inputs-1] — required by img2trajvid
        input_indices = list(range(num_inputs))

        # Infer anchor stats for two-pass
        num_anchors = infer_prior_stats(
            T_base, num_inputs,
            num_total_frames=num_targets,
            version_dict=version_dict,
        )
        assert isinstance(num_anchors, int)

        # Anchors evenly spaced among target positions (indices num_inputs..num_frames_total-1)
        anchor_indices = np.linspace(
            num_inputs, num_inputs + num_targets - 1, num_anchors
        ).tolist()
        anchor_c2ws = all_c2ws[[round(i) for i in anchor_indices]]
        anchor_Ks = all_Ks_pixel[[round(i) for i in anchor_indices]]

        all_imgs_np = (
            F.pad(input_imgs, (0, 0, 0, 0, 0, 0, 0, num_targets), value=0.0).numpy()
            * 255.0
        ).astype(np.uint8)

        image_cond = {
            "img": all_imgs_np,
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
        }
        camera_cond = {
            "c2w": all_c2ws.float(),
            "K": all_Ks_pixel.float(),
            "input_indices": list(range(num_inputs + num_targets)),
        }

        return version_dict, image_cond, camera_cond, anchor_c2ws, anchor_Ks

    def _seva_denoise_step(self, x, sigma, c, uc, T, guider, scale,
                           c2w, K, input_frame_mask):
        """One SEVA denoising step using the MultiviewCFG guider."""
        s_in = x.new_ones([x.shape[0]])
        sigma_batch = s_in * sigma
        x_cat, s_cat, c_merged = guider.prepare_inputs(x, sigma_batch, c, uc)
        denoised = self.denoiser(
            self.seva_model, x_cat, s_cat, c_merged,
            num_frames=T,
        )
        denoised = guider(
            denoised, sigma_batch, scale,
            c2w=c2w, K=K, input_frame_mask=input_frame_mask,
        )
        return denoised

    def _sd_x0(
        self,
        x,
        t,
        text_cond,
        text_uncond,
        w2c,
        all_Ks,
        input_indices,
        sd_ref_latents,
    ):
        """Predict x0 from PoseSD using view-packing and cross-view attention."""
        alpha_bar = self.sd_alphas[t]
        h, w = x.shape[-2], x.shape[-1]
        N = x.shape[0]

        ref1_lats, ref2_lats = [], []
        pl_ref1s, pl_ref2s, pl_tgts = [], [], []

        for j in range(N):
            closest = sorted(input_indices, key=lambda idx: abs(idx - j))[:2]
            if len(closest) == 1:
                closest = [closest[0], closest[0]]
            r1_idx, r2_idx = closest

            ref1_lats.append(sd_ref_latents[input_indices.index(r1_idx)])
            ref2_lats.append(sd_ref_latents[input_indices.index(r2_idx)])

            anchor = w2c[r1_idx]
            pl_ref1s.append(get_plucker_coordinates(
                anchor, w2c[r1_idx:r1_idx+1],
                all_Ks[r1_idx:r1_idx+1].clone(), [h, w],
            )[0])
            pl_ref2s.append(get_plucker_coordinates(
                anchor, w2c[r2_idx:r2_idx+1],
                all_Ks[r2_idx:r2_idx+1].clone(), [h, w],
            )[0])
            pl_tgts.append(get_plucker_coordinates(
                anchor, w2c[j:j+1],
                all_Ks[j:j+1].clone(), [h, w],
            )[0])

        ref1_lat = torch.stack(ref1_lats)
        ref2_lat = torch.stack(ref2_lats)
        pl_ref1 = torch.stack(pl_ref1s)
        pl_ref2 = torch.stack(pl_ref2s)
        pl_tgt = torch.stack(pl_tgts)

        t_cond = text_cond.expand(N, -1, -1)
        t_uncond = text_uncond.expand(N, -1, -1)
        t_b = torch.full((N,), int(t), device=self.device, dtype=torch.long)

        packed_c = self.pose_sd._pack_views(
            ref1_lat, ref2_lat, x, pl_ref1, pl_ref2, pl_tgt,
        )
        eps_c = self.pose_sd._predict_target_eps(packed_c, t_b, t_cond, N)

        keep_none = torch.zeros(N, device=self.device, dtype=torch.bool)
        packed_u = self.pose_sd._pack_views(
            ref1_lat, ref2_lat, x, pl_ref1, pl_ref2, pl_tgt, keep_none,
        )
        eps_u = self.pose_sd._predict_target_eps(packed_u, t_b, t_uncond, N)

        eps = self._cfg_combine(eps_c, eps_u, self.cfg_sd)
        return self._x0_from_eps(x, eps, alpha_bar)

    def _encode_sd_refs_from_preprocessed(self, input_imgs_01: torch.Tensor) -> torch.Tensor:
        latents = []
        for i in range(input_imgs_01.shape[0]):
            tensor = (
                input_imgs_01[i : i + 1].permute(0, 3, 1, 2).to(self.device) * 2.0 - 1.0
            )
            latents.append(self.pose_sd.encode_image(tensor))
        return torch.cat(latents, dim=0)

    def _sigma_to_sd_t(self, sigma):
        if sigma <= 0:
            return 0
        sigma_sd = sigma / np.exp(2.4)
        alpha_bar = 1.0 / (1.0 + sigma_sd.item() ** 2)
        return (self.sd_alphas - alpha_bar).abs().argmin().item()

    def _decode_frames(self, latents):
        results = []
        for j in range(latents.shape[0]):
            with self._autocast_context():
                decoded = self.ae.decode(latents[j:j+1], 1)
            out = ((decoded.clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(1, 2, 0).cpu().numpy()
            results.append(Image.fromarray(out))
        return results

    def sample_seva_only(self, images, num_frames=21, seed=42, save_path=None):
        """SEVA-only sampling using the full demo pipeline with DUSt3R + two-pass."""
        torch.manual_seed(seed)
        input_images, _ = self._parse_images(images)
        img_paths = self._images_to_paths(input_images)

        # DUSt3R needs gradients — run before inference_mode
        version_dict, image_cond, camera_cond, anchor_c2ws, anchor_Ks = (
            self._build_run_one_scene_args(img_paths, num_frames)
        )

        if save_path is None:
            save_path = tempfile.mkdtemp(prefix="seva_")

        # run_one_scene uses its own inference_mode internally
        results_gen = run_one_scene(
            task="img2trajvid",
            version_dict=version_dict,
            model=self.seva_model,
            ae=self.ae,
            conditioner=self.clip_cond,
            denoiser=self.denoiser,
            image_cond=image_cond,
            camera_cond=camera_cond,
            save_path=save_path,
            use_traj_prior=True,
            traj_prior_c2ws=anchor_c2ws,
            traj_prior_Ks=anchor_Ks,
            seed=seed,
            gradio=True,
        )

        # run_one_scene is a generator, consume it
        video_path = None
        for vp in results_gen:
            video_path = vp

        # Clean up temp image files
        for p in img_paths:
            if p.startswith(tempfile.gettempdir()):
                os.unlink(p)

        return video_path

    def sample(self, images, prompt, num_frames=21, seed=42):
        """Coupled SEVA + PoseSD sampling with DUSt3R poses + MultiviewCFG."""
        torch.manual_seed(seed)
        input_images, _ = self._parse_images(images)
        img_paths = self._images_to_paths(input_images)
        num_inputs = len(img_paths)

        # Match demo_gr.py preprocessing to get reliable cameras/intrinsics.
        input_imgs_01, input_Ks, input_c2ws, (W, H) = self._preprocess_inputs_for_seva(
            img_paths
        )

        latent_h, latent_w = H // 8, W // 8

        # Build unified trajectory: num_frames total, inputs at keyframe positions
        all_c2ws, all_Ks, input_indices = self._build_unified_trajectory(
            input_c2ws, input_Ks, num_frames, (W, H)
        )

        with torch.inference_mode():
            N = num_frames

            # Encode input images for SEVA conditioning
            imgs = torch.zeros(N, 3, H, W)
            input_imgs_nchw = input_imgs_01.permute(0, 3, 1, 2) * 2.0 - 1.0
            for i, idx in enumerate(input_indices):
                imgs[idx] = input_imgs_nchw[i]

            # Camera centering + normalization (replicates get_value_dict logic)
            c2w = to_hom_pose(all_c2ws.float())
            ref_c2ws = c2w.clone()
            camera_dist_2med = torch.norm(
                ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
                dim=-1,
            )
            valid_mask = camera_dist_2med <= torch.clamp(
                torch.quantile(camera_dist_2med, 0.97) * 10, max=1e6,
            )
            c2w[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)
            w2c = torch.linalg.inv(c2w)
            camera_dists = c2w[:, :3, 3].clone()
            t_scale = (
                self.camera_scale
                if torch.isclose(torch.norm(camera_dists[0]), torch.zeros(1), atol=1e-5).any()
                else (self.camera_scale / torch.norm(camera_dists[0]))
            )
            w2c[:, :3, 3] *= t_scale
            c2w[:, :3, 3] *= t_scale

            pluckers = get_plucker_coordinates(
                extrinsics_src=w2c[0],
                extrinsics=w2c,
                intrinsics=all_Ks.float().clone(),
                target_size=(latent_h, latent_w),
            ).to(self.device)

            input_masks = torch.zeros(N, dtype=torch.bool, device=self.device)
            for idx in input_indices:
                input_masks[idx] = True
            cond_imgs = imgs.to(self.device)
            c2w_guider = c2w.to(self.device)
            K_guider = all_Ks.to(self.device)
            T = N

            with self._autocast_context():
                latents = F.pad(
                    self.ae.encode(cond_imgs[input_masks], 1),
                    (0, 0, 0, 0, 0, 1), value=1.0,
                )
                c_crossattn = repeat(
                    self.clip_cond(cond_imgs[input_masks]).mean(0),
                    "d -> n 1 d", n=T,
                )

            uc_crossattn = torch.zeros_like(c_crossattn)
            c_replace = latents.new_zeros(T, *latents.shape[1:])
            c_replace[input_masks] = latents
            uc_replace = torch.zeros_like(c_replace)

            c_concat = torch.cat([
                repeat(input_masks, "n -> n 1 h w",
                       h=pluckers.shape[2], w=pluckers.shape[3]),
                pluckers,
            ], 1)
            uc_concat = torch.cat([
                pluckers.new_zeros(T, 1, *pluckers.shape[-2:]),
                pluckers,
            ], 1)

            c = {
                "crossattn": c_crossattn, "replace": c_replace,
                "concat": c_concat, "dense_vector": pluckers,
            }
            uc = {
                "crossattn": uc_crossattn, "replace": uc_replace,
                "concat": uc_concat, "dense_vector": pluckers,
            }

            # PoseSD w2c for Plucker computation (from centered+scaled c2ws)
            w2c_device = w2c.to(self.device)
            sd_all_Ks = all_Ks.to(self.device)

            # SD reference encoding and text embeddings
            sd_ref_latents = self._encode_sd_refs_from_preprocessed(input_imgs_01)
            text_cond = self.pose_sd.get_text_embeddings([prompt])
            text_uncond = self.pose_sd.null_text_emb

            # MultiviewCFG guider
            guider = MultiviewCFG(cfg_min=self.cfg_min)

            # Sigma schedule
            sigmas = self.discretization(self.num_steps, device=self.device)
            noise = torch.randn(T, 4, latent_h, latent_w, device=self.device)
            x_seva = noise * torch.sqrt(1.0 + sigmas[0] ** 2)
            x_sd = noise.clone()

            for i in range(len(sigmas) - 1):
                sigma, sigma_next = sigmas[i], sigmas[i + 1]
                sd_t = self._sigma_to_sd_t(sigma)
                sd_t_next = self._sigma_to_sd_t(sigma_next)

                # SEVA x0 with MultiviewCFG
                with self._autocast_context():
                    x0_seva = self._seva_denoise_step(
                        x_seva, sigma, c, uc, T, guider, self.cfg_seva,
                        c2w_guider, K_guider, input_masks,
                    )

                # PoseSD x0
                x0_sd = self._sd_x0(
                    x_sd, sd_t, text_cond, text_uncond,
                    w2c_device, sd_all_Ks, input_indices, sd_ref_latents,
                )

                # Coupling
                grad = self.coupling_strength * (x0_seva - x0_sd)
                x0_seva_c = x0_seva - grad
                x0_sd_c = x0_sd + grad

                # SEVA Euler step
                sigma_hat = sigma + 1e-6
                d = to_d(x_seva, append_dims(sigma_hat, x_seva.ndim).squeeze(), x0_seva_c)
                x_seva = x_seva + append_dims(sigma_next - sigma_hat, x_seva.ndim) * d

                # SD DDPM step
                if sd_t_next > 0:
                    alpha_next = self.sd_alphas[sd_t_next]
                    x_sd = alpha_next.sqrt() * x0_sd_c + (1 - alpha_next).sqrt() * torch.randn_like(x_sd)
                else:
                    x_sd = x0_sd_c

        # Clean up temp files
        for p in img_paths:
            if p.startswith(tempfile.gettempdir()):
                os.unlink(p)

        return self._decode_frames(x_seva)


if __name__ == "__main__":
    import argparse
    import imageio

    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Image paths")
    p.add_argument("--pretrained", default="manojb/stable-diffusion-2-1-base")
    p.add_argument("--pose-sd-checkpoint",
                   default="/vulcanscratch/ltahboub/CoupledSceneSampling/checkpoints/pose_sd_v1/unet_latest.pt",
                   help="Path to trained UNet checkpoint (.pt)")
    p.add_argument("--prompt", default="")
    p.add_argument("--frames", type=int, default=21)
    p.add_argument("--output", default="output.mp4")
    p.add_argument("--coupling", type=float, default=0.2)
    p.add_argument("--cfg-seva", type=float, default=3.0)
    p.add_argument("--cfg-sd", type=float, default=7.5)
    p.add_argument("--cfg-min", type=float, default=1.2)
    p.add_argument("--camera-scale", type=float, default=2.0)
    p.add_argument("--chunk-strategy", default="interp-gt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare-seva", action="store_true")
    args = p.parse_args()

    # Input images are just paths now (no more :frame_idx needed)
    images = [(img_path, i) for i, img_path in enumerate(args.inputs)]

    sampler = CoupledDiffusionSampler(
        args.pretrained,
        pose_sd_checkpoint=args.pose_sd_checkpoint,
        coupling_strength=args.coupling,
        cfg_seva=args.cfg_seva,
        cfg_sd=args.cfg_sd,
        cfg_min=args.cfg_min,
        camera_scale=args.camera_scale,
        chunk_strategy=args.chunk_strategy,
    )

    results = sampler.sample(images, args.prompt, args.frames, args.seed)
    imageio.mimsave(args.output, [np.array(img) for img in results], fps=30)
    print(f"Saved coupled: {args.output}")

    if args.compare_seva:
        seva_output = args.output.replace(".mp4", "_seva_only.mp4")
        video_path = sampler.sample_seva_only(images, args.frames, args.seed)
        if video_path:
            import shutil
            shutil.copy(video_path, seva_output)
            print(f"Saved SEVA-only: {seva_output}")
        else:
            print("SEVA-only sampling returned no video")
