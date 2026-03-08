import argparse
import contextlib
import os
import os.path as osp
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat

from run_seva_keyframe_video import (
    build_target_trajectory_from_input_keyframes,
    preprocess_advanced,
    run as run_seva_only,
)
from seva.geometry import get_plucker_coordinates, to_hom_pose
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
from seva.utils import load_model as load_seva_model


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from css.models.EMA import load_pose_sd_checkpoint
from css.models.pose_sd import PoseSD


class CoupledDiffusionRunner:
    def __init__(
        self,
        pose_sd_pretrained: str,
        pose_sd_checkpoint: str | None,
        device: str,
        use_half: bool = True,
        offload_pose_unet: bool = True,
        offload_seva_model: bool = True,
    ):
        self.device = device
        self.use_half = use_half and str(device).startswith("cuda")
        self.sample_dtype = torch.float16 if self.use_half else torch.float32
        self.offload_pose_unet = offload_pose_unet
        self.offload_seva_model = offload_seva_model
        self.seva_model = SGMWrapper(load_seva_model(device="cpu", verbose=True).eval()).to(device)
        self.ae = AutoEncoder(chunk_size=1).to(device)
        self.conditioner = CLIPConditioner().to(device)
        self.denoiser = DiscreteDenoiser(num_idx=1000, device=device)
        self.discretization = DDPMDiscretization()
        self.dust3r = Dust3rPipeline(device=device)

        self.pose_sd = PoseSD(pretrained_model=pose_sd_pretrained, device=device)
        if pose_sd_checkpoint:
            load_pose_sd_checkpoint(self.pose_sd, pose_sd_checkpoint, device)
        self.pose_sd.eval()
        self.sd_alphas = self.pose_sd.scheduler.alphas_cumprod.to(device)
        if self.use_half:
            self.seva_model.half()
            self.ae.half()
            self.conditioner.half()
            self.pose_sd.unet.half()
            self.pose_sd.vae.half()
            self.pose_sd.text_encoder.half()
        if self.offload_pose_unet:
            self.pose_sd.unet.to("cpu")
        if self.offload_seva_model:
            self.seva_model.to("cpu")
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def _autocast_context(self):
        return (
            torch.autocast("cuda")
            if str(self.device).startswith("cuda")
            else contextlib.nullcontext()
        )

    @staticmethod
    def _x0_from_eps(latent, eps, alpha_bar):
        alpha_bar = alpha_bar.to(device=latent.device, dtype=latent.dtype)
        sqrt_alpha = alpha_bar.sqrt()
        sqrt_one_minus = (1.0 - alpha_bar).clamp_min(0.0).sqrt()
        return (latent - sqrt_one_minus * eps) / sqrt_alpha

    @staticmethod
    def _cfg(cond, uncond, scale):
        if scale <= 1.0:
            return cond
        return uncond + scale * (cond - uncond)

    def _sigma_to_sd_t(self, sigma):
        if sigma <= 0:
            return 0
        sigma_sd = sigma / np.exp(2.4)
        alpha_bar = 1.0 / (1.0 + sigma_sd.item() ** 2)
        return (self.sd_alphas - alpha_bar).abs().argmin().item()

    @staticmethod
    def _build_ref_pairs(all_c2ws, input_indices):
        input_positions = all_c2ws[input_indices, :3, 3]
        pairs = []
        for j in range(all_c2ws.shape[0]):
            dists = torch.norm(input_positions - all_c2ws[j, :3, 3], dim=-1)
            closest = torch.argsort(dists)[: min(2, len(input_indices))]
            if closest.numel() == 1:
                closest = torch.cat([closest, closest], dim=0)
            pairs.append((input_indices[int(closest[0])], input_indices[int(closest[1])]))
        return pairs

    @staticmethod
    def _resize_preprocessed(input_imgs, input_Ks, short_side):
        W = int(input_imgs.shape[2])
        H = int(input_imgs.shape[1])
        if min(H, W) <= short_side:
            return input_imgs, input_Ks, (W, H)
        scale = short_side / float(min(H, W))
        new_h = int(round((H * scale) / 64) * 64)
        new_w = int(round((W * scale) / 64) * 64)
        resized = F.interpolate(
            input_imgs.permute(0, 3, 1, 2),
            size=(new_h, new_w),
            mode="area",
            antialias=False,
        ).permute(0, 2, 3, 1)
        # input_Ks are normalized intrinsics, unchanged by resize.
        return resized, input_Ks, (new_w, new_h)

    def _build_seva_conditioning(
        self,
        cond_imgs,
        all_c2ws,
        all_Ks,
        input_indices,
        camera_scale,
        cfg_min,
    ):
        N, H, W = cond_imgs.shape[0], cond_imgs.shape[2], cond_imgs.shape[3]
        latent_h, latent_w = H // 8, W // 8

        c2w = to_hom_pose(all_c2ws.float()).clone()
        ref_c2ws = c2w.clone()
        camera_dist_2med = torch.norm(
            ref_c2ws[:, :3, 3] - ref_c2ws[:, :3, 3].median(0, keepdim=True).values,
            dim=-1,
        )
        valid_mask = camera_dist_2med <= torch.clamp(
            torch.quantile(camera_dist_2med, 0.97) * 10, max=1e6
        )
        c2w[:, :3, 3] -= ref_c2ws[valid_mask, :3, 3].mean(0, keepdim=True)
        w2c = torch.linalg.inv(c2w)
        camera_dists = c2w[:, :3, 3].clone()
        t_scale = (
            camera_scale
            if torch.isclose(torch.norm(camera_dists[0]), torch.zeros(1), atol=1e-5).any()
            else (camera_scale / torch.norm(camera_dists[0]))
        )
        w2c[:, :3, 3] *= t_scale
        c2w[:, :3, 3] *= t_scale

        pluckers = get_plucker_coordinates(
            extrinsics_src=w2c[0],
            extrinsics=w2c,
            intrinsics=all_Ks.float().clone(),
            target_size=(latent_h, latent_w),
        ).to(self.device, dtype=cond_imgs.dtype)

        input_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        input_mask[input_indices] = True
        cond_imgs = cond_imgs.to(self.device)

        with self._autocast_context():
            latents = F.pad(
                self.ae.encode(cond_imgs[input_mask], 1),
                (0, 0, 0, 0, 0, 1),
                value=1.0,
            )
            c_crossattn = repeat(
                self.conditioner(cond_imgs[input_mask]).mean(0),
                "d -> n 1 d",
                n=N,
            )

        uc_crossattn = torch.zeros_like(c_crossattn)
        c_replace = latents.new_zeros(N, *latents.shape[1:])
        c_replace[input_mask] = latents
        uc_replace = torch.zeros_like(c_replace)
        c_concat = torch.cat(
            [
                repeat(input_mask, "n -> n 1 h w", h=pluckers.shape[2], w=pluckers.shape[3]),
                pluckers,
            ],
            1,
        )
        uc_concat = torch.cat(
            [pluckers.new_zeros(N, 1, *pluckers.shape[-2:]), pluckers],
            1,
        )

        c = {
            "crossattn": c_crossattn,
            "replace": c_replace,
            "concat": c_concat,
            "dense_vector": pluckers,
        }
        uc = {
            "crossattn": uc_crossattn,
            "replace": uc_replace,
            "concat": uc_concat,
            "dense_vector": pluckers,
        }
        guider = MultiviewCFG(cfg_min=cfg_min)
        return c, uc, guider, input_mask, c2w.to(self.device), all_Ks.to(self.device), w2c.to(self.device)

    def _sd_x0(
        self,
        x,
        t,
        text_cond,
        text_uncond,
        w2c,
        all_Ks,
        ref_pairs,
        idx_to_ref,
        sd_ref_latents,
        cfg_sd,
        sd_chunk_size,
    ):
        if self.offload_pose_unet:
            self.pose_sd.unet.to(self.device)
            torch.cuda.empty_cache()
        alpha_bar = self.sd_alphas[t]
        h, w = x.shape[-2], x.shape[-1]
        N = x.shape[0]
        x0_chunks = []
        for start in range(0, N, sd_chunk_size):
            end = min(start + sd_chunk_size, N)
            x_chunk = x[start:end]
            C = x_chunk.shape[0]

            ref1_lats, ref2_lats = [], []
            pl_ref1s, pl_ref2s, pl_tgts = [], [], []
            for j in range(start, end):
                r1_idx, r2_idx = ref_pairs[j]
                ref1_lats.append(sd_ref_latents[idx_to_ref[r1_idx]])
                ref2_lats.append(sd_ref_latents[idx_to_ref[r2_idx]])

                anchor = w2c[r1_idx]
                pl_ref1s.append(
                    get_plucker_coordinates(
                        extrinsics_src=anchor,
                        extrinsics=w2c[r1_idx : r1_idx + 1],
                        intrinsics=all_Ks[r1_idx : r1_idx + 1].clone(),
                        target_size=[h, w],
                    )[0]
                )
                pl_ref2s.append(
                    get_plucker_coordinates(
                        extrinsics_src=anchor,
                        extrinsics=w2c[r2_idx : r2_idx + 1],
                        intrinsics=all_Ks[r2_idx : r2_idx + 1].clone(),
                        target_size=[h, w],
                    )[0]
                )
                pl_tgts.append(
                    get_plucker_coordinates(
                        extrinsics_src=anchor,
                        extrinsics=w2c[j : j + 1],
                        intrinsics=all_Ks[j : j + 1].clone(),
                        target_size=[h, w],
                    )[0]
                )

            ref1_lat = torch.stack(ref1_lats)
            ref2_lat = torch.stack(ref2_lats)
            pl_ref1 = torch.stack(pl_ref1s).to(x_chunk.dtype)
            pl_ref2 = torch.stack(pl_ref2s).to(x_chunk.dtype)
            pl_tgt = torch.stack(pl_tgts).to(x_chunk.dtype)

            t_b = torch.full((C,), int(t), device=self.device, dtype=torch.long)
            t_cond = text_cond.expand(C, -1, -1)
            t_uncond = text_uncond.expand(C, -1, -1)
            with self._autocast_context():
                packed_c = self.pose_sd._pack_views(
                    ref1_lat, ref2_lat, x_chunk, pl_ref1, pl_ref2, pl_tgt
                )
                eps_c = self.pose_sd._predict_target_eps(packed_c, t_b, t_cond, C)
                keep_none = torch.zeros(C, device=self.device, dtype=torch.bool)
                packed_u = self.pose_sd._pack_views(
                    ref1_lat, ref2_lat, x_chunk, pl_ref1, pl_ref2, pl_tgt, keep_none
                )
                eps_u = self.pose_sd._predict_target_eps(packed_u, t_b, t_uncond, C)
            eps = self._cfg(eps_c, eps_u, cfg_sd)
            x0_chunks.append(self._x0_from_eps(x_chunk, eps, alpha_bar))
        out = torch.cat(x0_chunks, 0)
        if self.offload_pose_unet:
            self.pose_sd.unet.to("cpu")
            torch.cuda.empty_cache()
        return out

    def sample_coupled(
        self,
        img_paths,
        prompt,
        seed,
        coupling_strength,
        cfg_seva,
        cfg_sd,
        cfg_min,
        num_steps,
        camera_scale,
        fps,
        transition_sec,
        sd_chunk_size,
        max_total_frames,
        short_side,
    ):
        torch.manual_seed(seed)
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        # DUSt3R camera optimization performs backward passes internally.
        # Force-enable grads here in case caller wrapped us in inference/no-grad mode.
        with torch.inference_mode(False), torch.enable_grad():
            input_imgs, input_Ks, input_c2ws, (W, H) = preprocess_advanced(
                img_paths, self.dust3r
            )
        if max_total_frames is not None and max_total_frames > 0:
            max_total_frames = max(2, int(max_total_frames))
            if input_imgs.shape[0] > max_total_frames:
                keep = (
                    np.linspace(0, input_imgs.shape[0] - 1, max_total_frames)
                    .round()
                    .astype(int)
                )
                input_imgs = input_imgs[keep]
                input_Ks = input_Ks[keep]
                input_c2ws = input_c2ws[keep]
        input_imgs, input_Ks, (W, H) = self._resize_preprocessed(
            input_imgs, input_Ks, short_side
        )
        target_c2ws, target_Ks = build_target_trajectory_from_input_keyframes(
            input_c2ws=input_c2ws,
            input_Ks=input_Ks,
            input_wh=(W, H),
            fps=fps,
            transition_sec=transition_sec,
        )

        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            num_inputs = input_imgs.shape[0]
            if max_total_frames is not None and max_total_frames > num_inputs:
                max_targets = max_total_frames - num_inputs
                if target_c2ws.shape[0] > max_targets:
                    keep = np.linspace(0, target_c2ws.shape[0] - 1, max_targets).round().astype(int)
                    target_c2ws = target_c2ws[keep]
                    target_Ks = target_Ks[keep]

            input_indices = list(range(num_inputs))
            all_c2ws = torch.cat([input_c2ws, target_c2ws], 0)
            all_Ks = torch.cat([input_Ks, target_Ks], 0)
            N = all_c2ws.shape[0]
            latent_h, latent_w = H // 8, W // 8

            cond_imgs = torch.zeros(N, 3, H, W, dtype=self.sample_dtype)
            cond_imgs[:num_inputs] = (
                input_imgs.permute(0, 3, 1, 2).to(dtype=self.sample_dtype) * 2.0 - 1.0
            )
            c, uc, guider, input_mask, c2w_guider, K_guider, w2c = self._build_seva_conditioning(
                cond_imgs,
                all_c2ws,
                all_Ks,
                input_indices,
                camera_scale=camera_scale,
                cfg_min=cfg_min,
            )

            with self._autocast_context():
                sd_ref_latents = self.pose_sd.encode_image(
                    (
                        input_imgs.permute(0, 3, 1, 2).to(self.device, dtype=self.sample_dtype)
                        * 2.0
                        - 1.0
                    )
                )
            text_cond = self.pose_sd.get_text_embeddings([prompt])
            text_uncond = self.pose_sd.null_text_emb
            # Offload PoseSD components not used during iterative denoising.
            self.pose_sd.vae.to("cpu")
            self.pose_sd.text_encoder.to("cpu")
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()

            # Offload SEVA helper modules not used in iterative denoising.
            self.ae.to("cpu")
            self.conditioner.to("cpu")
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()

            ref_pairs = self._build_ref_pairs(all_c2ws, input_indices)
            idx_to_ref = {idx: i for i, idx in enumerate(input_indices)}

            sigmas = self.discretization(num_steps, device=self.device)
            noise = torch.randn(N, 4, latent_h, latent_w, device=self.device, dtype=self.sample_dtype)
            x_seva = noise * torch.sqrt(1.0 + sigmas[0] ** 2)
            x_sd = noise.clone()

            for i in range(len(sigmas) - 1):
                sigma, sigma_next = sigmas[i], sigmas[i + 1]

                if self.offload_seva_model:
                    self.seva_model.to(self.device)
                    torch.cuda.empty_cache()
                sigma_batch = x_seva.new_ones([N]) * sigma
                x_cat, s_cat, c_merged = guider.prepare_inputs(x_seva, sigma_batch, c, uc)
                with self._autocast_context():
                    denoised = self.denoiser(
                        self.seva_model, x_cat, s_cat, c_merged, num_frames=N
                    )
                    x0_seva = guider(
                        denoised,
                        sigma_batch,
                        cfg_seva,
                        c2w=c2w_guider,
                        K=K_guider,
                        input_frame_mask=input_mask,
                    )
                if self.offload_seva_model:
                    self.seva_model.to("cpu")
                    torch.cuda.empty_cache()

                sd_t = self._sigma_to_sd_t(sigma)
                x0_sd = self._sd_x0(
                    x_sd,
                    sd_t,
                    text_cond,
                    text_uncond,
                    w2c,
                    K_guider,
                    ref_pairs,
                    idx_to_ref,
                    sd_ref_latents,
                    cfg_sd,
                    sd_chunk_size,
                )

                grad = coupling_strength * (x0_seva - x0_sd)
                x0_seva_c = x0_seva - grad
                x0_sd_c = x0_sd + grad

                sigma_hat = sigma + 1e-6
                d = to_d(x_seva, append_dims(sigma_hat, x_seva.ndim).squeeze(), x0_seva_c)
                x_seva = x_seva + append_dims(sigma_next - sigma_hat, x_seva.ndim) * d

                sd_t_next = self._sigma_to_sd_t(sigma_next)
                if sd_t_next > 0:
                    alpha_next = self.sd_alphas[sd_t_next]
                    x_sd = alpha_next.sqrt() * x0_sd_c + (1 - alpha_next).sqrt() * torch.randn_like(x_sd)
                else:
                    x_sd = x0_sd_c

            frames = []
            self.ae.to(self.device)
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
            for j in range(N):
                with self._autocast_context():
                    decoded = self.ae.decode(x_seva[j : j + 1], 1)
                frame = ((decoded.clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(1, 2, 0).cpu().numpy()
                frames.append(frame)
            return np.stack(frames, 0)
        finally:
            torch.set_grad_enabled(prev_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["seva", "coupled"], default="coupled")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--save-root", default="work_dirs/coupled_diffusion")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--prompt", default="")
    parser.add_argument("--pose-sd-pretrained", default="manojb/stable-diffusion-2-1-base")
    parser.add_argument("--pose-sd-checkpoint", default=None)
    parser.add_argument("--coupling-strength", type=float, default=0.2)
    parser.add_argument("--cfg-seva", type=float, default=3.0)
    parser.add_argument("--cfg-sd", type=float, default=7.5)
    parser.add_argument("--cfg-min", type=float, default=1.2)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--camera-scale", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--transition-sec", type=float, default=1.5)
    parser.add_argument("--sd-chunk-size", type=int, default=8)
    parser.add_argument("--max-total-frames", type=int, default=64)
    parser.add_argument("--short-side", type=int, default=384)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--no-offload-pose-unet", action="store_true")
    parser.add_argument("--no-offload-seva-model", action="store_true")
    args = parser.parse_args()

    img_paths = [osp.abspath(p) for p in args.inputs]
    output_path = osp.abspath(args.output)
    os.makedirs(osp.dirname(output_path), exist_ok=True)

    if args.mode == "seva":
        final_video = run_seva_only(
            img_paths=img_paths,
            output_path=output_path,
            save_root=args.save_root,
            seed=args.seed,
            device=args.device,
        )
        print(final_video)
        return

    runner = CoupledDiffusionRunner(
        pose_sd_pretrained=args.pose_sd_pretrained,
        pose_sd_checkpoint=args.pose_sd_checkpoint,
        device=args.device,
        use_half=not args.fp32,
        offload_pose_unet=not args.no_offload_pose_unet,
        offload_seva_model=not args.no_offload_seva_model,
    )
    frames = runner.sample_coupled(
        img_paths=img_paths,
        prompt=args.prompt,
        seed=args.seed,
        coupling_strength=args.coupling_strength,
        cfg_seva=args.cfg_seva,
        cfg_sd=args.cfg_sd,
        cfg_min=args.cfg_min,
        num_steps=args.num_steps,
        camera_scale=args.camera_scale,
        fps=args.fps,
        transition_sec=args.transition_sec,
        sd_chunk_size=max(1, args.sd_chunk_size),
        max_total_frames=(None if args.max_total_frames <= 0 else args.max_total_frames),
        short_side=max(64, args.short_side),
    )
    iio.imwrite(
        output_path,
        frames.astype(np.uint8),
        fps=args.fps,
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    print(output_path)


if __name__ == "__main__":
    main()
