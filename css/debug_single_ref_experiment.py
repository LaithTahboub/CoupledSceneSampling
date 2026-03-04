"""Minimal single-reference debugging experiment for CAT3D-style conditioning.

This script isolates a small 2D experiment:
- one reference image per sample,
- one nearby target image,
- no Plucker rays,
- cross-view attention over [target, reference] views.

It is designed to quickly validate whether the model can generate diverse
neighborhood samples instead of copying a reference view.
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image, ImageDraw
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from css.data.colmap_reader import read_scene
from css.data.dataset import (
    build_cropped_scaled_intrinsics,
    clean_scene_prompt_name,
    load_image_name_set,
    load_image_tensor,
    name_allowed,
    name_excluded,
    read_scene_prompt_name,
)
from css.data.frustum_iou import (
    build_camera_frustum_geometry,
    compute_frustum_iou_from_geometries,
    compute_reference_depth,
)
from css.models.cross_view_attention import CrossViewAttention


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    if mixed_precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    if mixed_precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unknown mixed precision mode: {mixed_precision}")


def _read_lines(path: str | None) -> list[str]:
    if path is None:
        return []
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _index_scene_images(images_dir: Path) -> tuple[set[str], dict[str, str]]:
    rel_paths: list[str] = []
    for p in images_dir.rglob("*"):
        if p.is_file():
            rel_paths.append(p.relative_to(images_dir).as_posix())

    rel_set = set(rel_paths)
    by_basename: dict[str, list[str]] = {}
    for rel in rel_paths:
        by_basename.setdefault(Path(rel).name, []).append(rel)
    unique_basename = {k: v[0] for k, v in by_basename.items() if len(v) == 1}
    return rel_set, unique_basename


def _resolve_image_name(image_name: str, rel_set: set[str], unique_basename: dict[str, str]) -> str | None:
    norm = image_name.replace("\\", "/")
    if norm in rel_set:
        return norm
    return unique_basename.get(Path(norm).name)


def _camera_forward(c2w: np.ndarray) -> np.ndarray:
    f = c2w[:3, 2].astype(np.float64)
    n = float(np.linalg.norm(f))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return f / n


def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = R_a.T @ R_b
    cos_theta = (float(np.trace(R_rel)) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _effective_focal(K: np.ndarray) -> float:
    fx = max(float(K[0, 0]), 1e-9)
    fy = max(float(K[1, 1]), 1e-9)
    return float(np.sqrt(fx * fy))


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    return ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()


@dataclass
class PairRecord:
    scene_name: str
    images_dir: Path
    ref_name: str
    target_name: str
    prompt: str
    iou: float
    distance: float
    rot_deg: float


class SingleRefPairDataset(Dataset):
    """Builds (reference, target) image pairs with moderate overlap."""

    def __init__(
        self,
        scene_dirs: list[str],
        *,
        H: int = 512,
        W: int = 512,
        min_pair_iou: float = 0.18,
        max_pair_iou: float = 0.62,
        min_pair_distance: float = 0.20,
        max_pair_distance: float = 2.2,
        min_view_cos: float = 0.80,
        min_rotation_deg: float = 3.0,
        max_rotation_deg: float = 35.0,
        max_focal_ratio: float = 1.35,
        prefilter_topk: int = 64,
        targets_per_ref: int = 2,
        max_pairs_per_scene: int = 128,
        exclude_image_names: set[str] | None = None,
        target_include_image_names: set[str] | None = None,
        reference_include_image_names: set[str] | None = None,
        prompt_template: str = "a photo of {scene}",
    ):
        if min_pair_iou < 0.0 or max_pair_iou > 1.0 or min_pair_iou >= max_pair_iou:
            raise ValueError("Expected 0 <= min_pair_iou < max_pair_iou <= 1")
        if min_pair_distance < 0.0 or min_pair_distance >= max_pair_distance:
            raise ValueError("Expected 0 <= min_pair_distance < max_pair_distance")
        if min_rotation_deg < 0.0 or min_rotation_deg >= max_rotation_deg:
            raise ValueError("Expected 0 <= min_rotation_deg < max_rotation_deg")
        if targets_per_ref <= 0:
            raise ValueError("targets_per_ref must be >= 1")
        if max_pairs_per_scene <= 0:
            raise ValueError("max_pairs_per_scene must be >= 1")

        self.H = H
        self.W = W
        self.exclude_image_names = exclude_image_names
        self.target_include_image_names = target_include_image_names
        self.reference_include_image_names = reference_include_image_names
        self.prompt_template = prompt_template

        self.pairs: list[PairRecord] = []
        for scene_spec in scene_dirs:
            scene_dir = Path(scene_spec)
            scene_name = scene_dir.name
            prompt = self._build_prompt(scene_dir)
            scene_pairs = self._build_scene_pairs(
                scene_dir=scene_dir,
                scene_name=scene_name,
                prompt=prompt,
                min_pair_iou=min_pair_iou,
                max_pair_iou=max_pair_iou,
                min_pair_distance=min_pair_distance,
                max_pair_distance=max_pair_distance,
                min_view_cos=min_view_cos,
                min_rotation_deg=min_rotation_deg,
                max_rotation_deg=max_rotation_deg,
                max_focal_ratio=max_focal_ratio,
                prefilter_topk=max(1, prefilter_topk),
                targets_per_ref=targets_per_ref,
                max_pairs=max_pairs_per_scene,
            )
            self.pairs.extend(scene_pairs)

        if len(self.pairs) == 0:
            print("SingleRefPairDataset: 0 valid pairs found.")
            return

        unique_images = set()
        scene_names = set()
        for p in self.pairs:
            scene_names.add(p.scene_name)
            unique_images.add(str(p.images_dir / p.ref_name))
            unique_images.add(str(p.images_dir / p.target_name))

        print(
            "SingleRefPairDataset ready: "
            f"{len(self.pairs)} pairs, {len(unique_images)} unique images, {len(scene_names)} scenes"
        )

    def _build_prompt(self, scene_dir: Path) -> str:
        scene_text = clean_scene_prompt_name(read_scene_prompt_name(scene_dir))
        template = self.prompt_template.strip()
        if not template:
            return ""
        if "{scene}" in template:
            return template.format(scene=scene_text)
        return template

    def _build_scene_pairs(
        self,
        *,
        scene_dir: Path,
        scene_name: str,
        prompt: str,
        min_pair_iou: float,
        max_pair_iou: float,
        min_pair_distance: float,
        max_pair_distance: float,
        min_view_cos: float,
        min_rotation_deg: float,
        max_rotation_deg: float,
        max_focal_ratio: float,
        prefilter_topk: int,
        targets_per_ref: int,
        max_pairs: int,
    ) -> list[PairRecord]:
        cameras, images = read_scene(scene_dir)
        images_dir = scene_dir / "images"
        rel_set, unique_basename = _index_scene_images(images_dir)

        valid = []
        resolved_name_by_id: dict[int, str] = {}
        for img in sorted(images.values(), key=lambda x: x.id):
            resolved = _resolve_image_name(img.name, rel_set, unique_basename)
            if resolved is None:
                continue
            if name_excluded(self.exclude_image_names, scene_name, img.name):
                continue
            resolved_name_by_id[img.id] = resolved
            valid.append(img)

        refs = [img for img in valid if name_allowed(self.reference_include_image_names, scene_name, img.name)]
        tgts = [img for img in valid if name_allowed(self.target_include_image_names, scene_name, img.name)]

        print(
            f"{scene_name}: usable={len(valid)} refs={len(refs)} tgts={len(tgts)} "
            f"(iou in [{min_pair_iou:.2f}, {max_pair_iou:.2f}])"
        )
        if len(refs) == 0 or len(tgts) == 0:
            return []

        positions = {img.id: img.c2w[:3, 3].astype(np.float64) for img in valid}
        R_by_id = {img.id: img.c2w[:3, :3].astype(np.float64) for img in valid}
        K_by_id = {img.id: build_cropped_scaled_intrinsics(cameras[img.camera_id], self.H, self.W) for img in valid}
        forward_by_id = {img.id: _camera_forward(img.c2w) for img in valid}
        focal_by_id = {img_id: _effective_focal(K_by_id[img_id]) for img_id in K_by_id}

        d_ref = compute_reference_depth(positions)
        geom_by_id: dict[int, dict[str, object]] = {
            img.id: build_camera_frustum_geometry(img.c2w, K_by_id[img.id], self.H, self.W, d_ref)
            for img in valid
        }
        iou_cache: dict[tuple[int, int], float] = {}

        def pair_key(i_id: int, j_id: int) -> tuple[int, int]:
            return (i_id, j_id) if i_id < j_id else (j_id, i_id)

        def pair_iou(i_id: int, j_id: int) -> float:
            key = pair_key(i_id, j_id)
            if key not in iou_cache:
                iou_cache[key] = compute_frustum_iou_from_geometries(geom_by_id[key[0]], geom_by_id[key[1]])
            return iou_cache[key]

        def pair_metrics(i_id: int, j_id: int) -> tuple[float, float, float, float]:
            dist = float(np.linalg.norm(positions[i_id] - positions[j_id]))
            dir_cos = float(np.clip(np.dot(forward_by_id[i_id], forward_by_id[j_id]), -1.0, 1.0))
            rot_deg = _rotation_angle_deg(R_by_id[i_id], R_by_id[j_id])
            fi = max(focal_by_id[i_id], 1e-9)
            fj = max(focal_by_id[j_id], 1e-9)
            focal_ratio = max(fi / fj, fj / fi)
            return dist, dir_cos, rot_deg, float(focal_ratio)

        iou_mid = 0.5 * (min_pair_iou + max_pair_iou)
        dist_mid = 0.5 * (min_pair_distance + max_pair_distance)
        rot_mid = 0.5 * (min_rotation_deg + max_rotation_deg)
        dist_span = max(1e-6, max_pair_distance - min_pair_distance)
        rot_span = max(1e-6, max_rotation_deg - min_rotation_deg)
        iou_span = max(1e-6, max_pair_iou - min_pair_iou)

        pair_candidates: list[tuple[float, PairRecord]] = []
        for ref in refs:
            scored_targets: list[tuple[float, PairRecord]] = []
            for tgt in tgts:
                if tgt.id == ref.id:
                    continue
                dist, dir_cos, rot_deg, focal_ratio = pair_metrics(ref.id, tgt.id)
                if dist < min_pair_distance or dist > max_pair_distance:
                    continue
                if dir_cos < min_view_cos:
                    continue
                if rot_deg < min_rotation_deg or rot_deg > max_rotation_deg:
                    continue
                if focal_ratio > max_focal_ratio:
                    continue

                iou = pair_iou(ref.id, tgt.id)
                if iou < min_pair_iou or iou > max_pair_iou:
                    continue

                iou_term = 1.0 - abs(iou - iou_mid) / iou_span
                dist_term = 1.0 - abs(dist - dist_mid) / dist_span
                rot_term = 1.0 - abs(rot_deg - rot_mid) / rot_span

                score = (
                    1.40 * iou_term
                    + 0.75 * dist_term
                    + 0.40 * rot_term
                    + 0.30 * dir_cos
                )

                scored_targets.append(
                    (
                        score,
                        PairRecord(
                            scene_name=scene_name,
                            images_dir=images_dir,
                            ref_name=resolved_name_by_id[ref.id],
                            target_name=resolved_name_by_id[tgt.id],
                            prompt=prompt,
                            iou=float(iou),
                            distance=float(dist),
                            rot_deg=float(rot_deg),
                        ),
                    )
                )

            scored_targets.sort(key=lambda x: x[0], reverse=True)
            pair_candidates.extend(scored_targets[:prefilter_topk][:targets_per_ref])

        if len(pair_candidates) == 0:
            return []

        pair_candidates.sort(key=lambda x: x[0], reverse=True)

        selected: list[PairRecord] = []
        used_targets: set[str] = set()
        for _, candidate in pair_candidates:
            if candidate.target_name in used_targets:
                continue
            selected.append(candidate)
            used_targets.add(candidate.target_name)
            if len(selected) >= max_pairs:
                break

        if len(selected) < max_pairs:
            selected_set = {(p.ref_name, p.target_name) for p in selected}
            for _, candidate in pair_candidates:
                key = (candidate.ref_name, candidate.target_name)
                if key in selected_set:
                    continue
                selected.append(candidate)
                selected_set.add(key)
                if len(selected) >= max_pairs:
                    break

        if len(selected) > 0:
            ious = np.array([x.iou for x in selected], dtype=np.float64)
            dists = np.array([x.distance for x in selected], dtype=np.float64)
            print(
                f"{scene_name}: selected={len(selected)} "
                f"iou(mean={ious.mean():.3f},min={ious.min():.3f},max={ious.max():.3f}) "
                f"dist(mean={dists.mean():.3f})"
            )
        return selected

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        rec = self.pairs[idx]
        ref_img, _, _ = load_image_tensor(rec.images_dir, rec.ref_name, self.H, self.W)
        target_img, _, _ = load_image_tensor(rec.images_dir, rec.target_name, self.H, self.W)
        return {
            "ref_img": ref_img,
            "target_img": target_img,
            "prompt": rec.prompt,
            "scene_name": rec.scene_name,
            "ref_name": rec.ref_name,
            "target_name": rec.target_name,
            "pair_iou": torch.tensor(rec.iou, dtype=torch.float32),
            "pair_distance": torch.tensor(rec.distance, dtype=torch.float32),
            "pair_rot_deg": torch.tensor(rec.rot_deg, dtype=torch.float32),
        }


class PairRecordViewDataset(Dataset):
    """Lightweight view over precomputed PairRecord entries."""

    def __init__(self, pairs: list[PairRecord], *, H: int, W: int):
        self.pairs = pairs
        self.H = H
        self.W = W

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        rec = self.pairs[idx]
        ref_img, _, _ = load_image_tensor(rec.images_dir, rec.ref_name, self.H, self.W)
        target_img, _, _ = load_image_tensor(rec.images_dir, rec.target_name, self.H, self.W)
        return {
            "ref_img": ref_img,
            "target_img": target_img,
            "prompt": rec.prompt,
            "scene_name": rec.scene_name,
            "ref_name": rec.ref_name,
            "target_name": rec.target_name,
            "pair_iou": torch.tensor(rec.iou, dtype=torch.float32),
            "pair_distance": torch.tensor(rec.distance, dtype=torch.float32),
            "pair_rot_deg": torch.tensor(rec.rot_deg, dtype=torch.float32),
        }


def split_pair_records(
    pairs: list[PairRecord],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[PairRecord], list[PairRecord]]:
    if len(pairs) == 0:
        return [], []
    if val_ratio <= 0.0:
        return list(pairs), []

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_val = int(round(len(pairs) * val_ratio))
    n_val = min(max(1, n_val), len(pairs) - 1)

    val_idx = set(order[:n_val].tolist())
    train_pairs = [p for i, p in enumerate(pairs) if i not in val_idx]
    val_pairs = [p for i, p in enumerate(pairs) if i in val_idx]
    return train_pairs, val_pairs


def _constant_with_warmup(optimizer: torch.optim.Optimizer, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def _expand_conv_in(unet: UNet2DConditionModel, new_in_channels: int) -> None:
    old = unet.conv_in
    if old.in_channels == new_in_channels:
        return

    new = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_channels].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.config.in_channels = new_in_channels


class SingleRefPoseSD(nn.Module):
    """2-view CAT3D-style debug model: [target, ref], no Plucker channels."""

    def __init__(
        self,
        pretrained_model: str = "manojb/stable-diffusion-2-1-base",
        device: str = "cuda",
    ):
        super().__init__()
        if device == "cuda" and not torch.cuda.is_available():
            print("[single-ref] CUDA unavailable, using CPU.")
            device = "cpu"
        self.device = torch.device(device)

        self.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.eval()

        self.latent_ch = 4
        self.mask_ch = 1
        self.per_view_ch = self.latent_ch + self.mask_ch
        self.num_views = 2

        _expand_conv_in(self.unet, new_in_channels=self.per_view_ch)
        self.cross_view_attention = CrossViewAttention(num_views=self.num_views, include_high_res=False)
        self.cross_view_attention.attach(self.unet)

        self._text_cache_limit = 4096
        self._cache_null_embeddings()

    def configure_trainable(self, train_mode: str = "cond") -> None:
        if train_mode == "full":
            self.unet.requires_grad_(True)
            return
        if train_mode != "cond":
            raise ValueError(f"Unknown train_mode: {train_mode}")
        self.unet.requires_grad_(False)
        self.unet.conv_in.requires_grad_(True)
        for name, param in self.unet.named_parameters():
            if "attn" in name:
                param.requires_grad_(True)

    def configure_memory_optimizations(
        self,
        gradient_checkpointing: bool = False,
        xformers_attention: bool = False,
    ) -> None:
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        if xformers_attention:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[single-ref] xformers unavailable: {exc}")
        self.cross_view_attention.attach(self.unet)

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.unet.parameters() if p.requires_grad]

    def _cache_null_embeddings(self) -> None:
        with torch.no_grad():
            tokens = self.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            self.null_text_emb = self.text_encoder(tokens.input_ids.to(self.device))[0]
            self._text_emb_cache: dict[str, torch.Tensor] = {"": self.null_text_emb}

    @torch.no_grad()
    def get_text_embeddings(self, prompts: Sequence[str | None]) -> torch.Tensor:
        normalized = [p if p is not None else "" for p in prompts]
        missing = [p for p in dict.fromkeys(normalized) if p not in self._text_emb_cache]
        if missing:
            tokens = self.tokenizer(missing, padding="max_length", max_length=77, return_tensors="pt")
            embeds = self.text_encoder(tokens.input_ids.to(self.device))[0]
            for i, prompt in enumerate(missing):
                self._text_emb_cache[prompt] = embeds[i : i + 1]
            if len(self._text_emb_cache) > self._text_cache_limit:
                self._text_emb_cache = {"": self.null_text_emb}
        return torch.cat([self._text_emb_cache[p] for p in normalized], dim=0)

    @torch.no_grad()
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample

    def _repeat_text_for_views(self, text_emb: torch.Tensor) -> torch.Tensor:
        return text_emb.repeat_interleave(self.num_views, dim=0)

    def _pack_views(
        self,
        *,
        ref_lat: torch.Tensor,
        tgt_lat: torch.Tensor,
        ref_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, _, h, w = tgt_lat.shape
        dtype = tgt_lat.dtype
        device = tgt_lat.device

        m_ref = torch.ones((b, 1, h, w), device=device, dtype=dtype)
        m_tgt = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
        if ref_keep_mask is not None:
            keep = ref_keep_mask.to(device=device, dtype=dtype).view(b, 1, 1, 1)
            ref_lat = ref_lat * keep
            m_ref = m_ref * keep

        v_ref = torch.cat([ref_lat, m_ref], dim=1)
        v_tgt = torch.cat([tgt_lat, m_tgt], dim=1)
        views = torch.stack([v_tgt, v_ref], dim=1)
        return views.reshape(b * self.num_views, self.per_view_ch, h, w)

    def _extract_target_view(self, pred_views: torch.Tensor, batch_size: int) -> torch.Tensor:
        if pred_views.shape[0] != batch_size * self.num_views:
            raise ValueError(
                f"Unexpected packed batch: got {pred_views.shape[0]}, expected {batch_size * self.num_views}"
            )
        return pred_views.view(batch_size, self.num_views, *pred_views.shape[1:])[:, 0]

    def _predict_target_eps(
        self,
        packed_views: torch.Tensor,
        timesteps: torch.Tensor,
        text_emb: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        t_views = timesteps.repeat_interleave(self.num_views)
        text_views = self._repeat_text_for_views(text_emb)
        pred_views = self.unet(packed_views, t_views, encoder_hidden_states=text_views).sample
        return self._extract_target_view(pred_views, batch_size=batch_size)

    def training_step(
        self,
        batch: dict,
        *,
        cond_drop_prob: float,
        min_timestep: int,
        max_timestep: int | None,
        noise_offset: float,
        min_snr_gamma: float | None,
    ) -> torch.Tensor:
        ref_img = batch["ref_img"].to(self.device)
        target_img = batch["target_img"].to(self.device)
        prompts = batch["prompt"]
        b = target_img.shape[0]

        ref_lat = self.encode_image(ref_img)
        tgt_lat_clean = self.encode_image(target_img)

        noise = torch.randn_like(tgt_lat_clean)
        if noise_offset > 0:
            noise = noise + noise_offset * torch.randn(
                (b, noise.shape[1], 1, 1),
                device=self.device,
                dtype=noise.dtype,
            )

        total_t = int(self.scheduler.config.num_train_timesteps)
        lo = max(0, int(min_timestep))
        hi = total_t - 1 if max_timestep is None else min(total_t - 1, int(max_timestep))
        if hi < lo:
            raise ValueError(f"Invalid timestep bounds [{lo}, {hi}]")

        timesteps = torch.randint(lo, hi + 1, (b,), device=self.device, dtype=torch.long)
        tgt_lat_noisy = self.scheduler.add_noise(tgt_lat_clean, noise, timesteps)

        if isinstance(prompts, str):
            text_emb = self.get_text_embeddings([prompts]).expand(b, -1, -1)
        else:
            if len(prompts) != b:
                raise ValueError(f"Prompt count ({len(prompts)}) != batch size ({b})")
            text_emb = self.get_text_embeddings(prompts)

        ref_keep_mask = None
        if self.training and cond_drop_prob > 0:
            drop = torch.rand(b, device=self.device) < cond_drop_prob
            if drop.any():
                text_emb = text_emb.clone()
                text_emb[drop] = self.null_text_emb.expand(int(drop.sum().item()), -1, -1)
                ref_keep_mask = ~drop

        packed = self._pack_views(ref_lat=ref_lat, tgt_lat=tgt_lat_noisy, ref_keep_mask=ref_keep_mask)
        pred = self._predict_target_eps(packed, timesteps, text_emb, batch_size=b)

        prediction_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(tgt_lat_clean, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=(1, 2, 3))

        if min_snr_gamma is not None and min_snr_gamma > 0:
            alphas = self.scheduler.alphas_cumprod.to(device=self.device, dtype=loss.dtype)
            alpha_t = alphas[timesteps]
            snr = alpha_t / (1.0 - alpha_t).clamp_min(1e-8)
            gamma = torch.full_like(snr, float(min_snr_gamma))
            if prediction_type == "epsilon":
                weights = torch.minimum(snr, gamma) / snr.clamp_min(1e-8)
            else:
                weights = torch.minimum(snr, gamma) / (snr + 1.0).clamp_min(1e-8)
            loss = loss * weights

        return loss.mean()

    @torch.inference_mode()
    def sample(
        self,
        *,
        ref_img: torch.Tensor,
        prompt: str,
        num_steps: int,
        cfg_scale: float,
        seed: int,
    ) -> torch.Tensor:
        b = ref_img.shape[0]
        if b != 1:
            raise ValueError("sample() currently expects batch size 1 for deterministic seeding")

        ref_lat = self.encode_image(ref_img.to(self.device))
        text_cond = self.get_text_embeddings([prompt]).expand(b, -1, -1)
        text_uncond = self.null_text_emb.expand(b, -1, -1)

        self.scheduler.set_timesteps(num_steps)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        latent = torch.randn(
            ref_lat.shape,
            generator=generator,
            device=self.device,
            dtype=ref_lat.dtype,
        )

        for t in tqdm(self.scheduler.timesteps, desc="sampling", leave=False):
            t_val = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            t_b = torch.full((b,), t_val, device=self.device, dtype=torch.long)
            latent_in = self.scheduler.scale_model_input(latent, t)

            packed_cond = self._pack_views(ref_lat=ref_lat, tgt_lat=latent_in, ref_keep_mask=None)
            eps_cond = self._predict_target_eps(packed_cond, t_b, text_cond, batch_size=b)

            keep_none = torch.zeros((b,), device=self.device, dtype=torch.bool)
            packed_uncond = self._pack_views(ref_lat=ref_lat, tgt_lat=latent_in, ref_keep_mask=keep_none)
            eps_uncond = self._predict_target_eps(packed_uncond, t_b, text_uncond, batch_size=b)

            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            latent = self.scheduler.step(eps, t, latent).prev_sample

        return self.decode_latent(latent)


def save_multisample_grid(
    *,
    ref_img: torch.Tensor,
    target_img: torch.Tensor,
    generated: list[torch.Tensor],
    output_path: Path,
    prompt: str,
    title: str,
) -> None:
    tiles = [_to_uint8(ref_img), _to_uint8(target_img)] + [_to_uint8(g) for g in generated]
    grid = np.concatenate(tiles, axis=1)
    canvas_h = grid.shape[0] + 30
    canvas = Image.new("RGB", (grid.shape[1], canvas_h), color=(255, 255, 255))
    canvas.paste(Image.fromarray(grid), (0, 30))
    text = f"{title} | prompt: {' '.join(prompt.split())}"
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), text[:280], fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_sample_strip(ref_img: torch.Tensor, target_img: torch.Tensor, generated_img: torch.Tensor) -> np.ndarray:
    return np.concatenate(
        [
            _to_uint8(ref_img),
            _to_uint8(target_img),
            _to_uint8(generated_img),
        ],
        axis=1,
    )


@torch.inference_mode()
def log_train_sample_to_wandb(
    model: SingleRefPoseSD,
    dataset: PairRecordViewDataset,
    *,
    output_dir: Path,
    epoch: int,
    global_step: int,
    num_steps: int,
    cfg_scale: float,
    sample_seed: int,
) -> None:
    if len(dataset) == 0:
        return
    try:
        idx = int((epoch - 1) % len(dataset))
        item = dataset[idx]
        rec = dataset.pairs[idx]
        prompt = item["prompt"]
        ref = item["ref_img"].unsqueeze(0).to(model.device)
        generated = model.sample(
            ref_img=ref,
            prompt=prompt,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            seed=sample_seed,
        )[0].detach().cpu()
        strip = build_sample_strip(item["ref_img"], item["target_img"], generated)
        out_path = output_dir / "train_samples" / f"epoch_{epoch:03d}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(strip).save(out_path)
        caption = (
            f'epoch={epoch} idx={idx} scene={rec.scene_name} '
            f'ref={rec.ref_name} tgt={rec.target_name} iou={rec.iou:.3f}'
        )
        wandb.log({"samples/train": wandb.Image(strip, caption=caption)}, step=global_step)
    except Exception as exc:
        print(f"[warning] failed to log train sample at epoch {epoch}: {exc}")


@torch.inference_mode()
def evaluate_diversity(
    model: SingleRefPoseSD,
    dataset: SingleRefPairDataset | PairRecordViewDataset,
    *,
    output_dir: Path,
    num_pairs: int,
    num_samples: int,
    sample_seed_base: int,
    num_steps: int,
    cfg_scale: float,
) -> list[dict]:
    if len(dataset) == 0:
        return []

    rng = np.random.default_rng(sample_seed_base)
    pair_indices = rng.choice(len(dataset), size=min(num_pairs, len(dataset)), replace=False).tolist()

    rows: list[dict] = []
    for rank, idx in enumerate(pair_indices):
        batch = dataset[idx]
        ref = batch["ref_img"].unsqueeze(0).to(model.device)
        tgt = batch["target_img"]
        prompt = batch["prompt"]

        samples: list[torch.Tensor] = []
        sample_stack: list[torch.Tensor] = []
        for sample_i in range(num_samples):
            seed = sample_seed_base + rank * 1000 + sample_i
            out = model.sample(
                ref_img=ref,
                prompt=prompt,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                seed=seed,
            )[0].detach().cpu()
            samples.append(out)
            sample_stack.append(out)

        stack_t = torch.stack(sample_stack, dim=0)
        diversity_l1 = float((stack_t - stack_t.mean(dim=0, keepdim=True)).abs().mean().item())
        ref_copy_l1 = float((stack_t - ref[0].cpu().unsqueeze(0)).abs().mean().item())

        rec = dataset.pairs[idx]
        title = f"{rec.scene_name} | ref={rec.ref_name} | tgt={rec.target_name} | iou={rec.iou:.3f}"
        save_multisample_grid(
            ref_img=batch["ref_img"],
            target_img=tgt,
            generated=samples,
            output_path=output_dir / f"pair_{rank:03d}.png",
            prompt=prompt,
            title=title,
        )
        rows.append(
            {
                "dataset_index": int(idx),
                "scene_name": rec.scene_name,
                "ref_name": rec.ref_name,
                "target_name": rec.target_name,
                "pair_iou": float(rec.iou),
                "pair_distance": float(rec.distance),
                "pair_rot_deg": float(rec.rot_deg),
                "diversity_l1": diversity_l1,
                "ref_copy_l1": ref_copy_l1,
                "grid_path": str((output_dir / f"pair_{rank:03d}.png").resolve()),
            }
        )

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--scenes-file", type=str, default=None)
    p.add_argument("--output", type=str, required=True)

    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--xformers-attention", action="store_true")
    p.add_argument("--train-mode", choices=["cond", "full"], default="cond")
    p.add_argument("--pretrained-model", type=str, default="manojb/stable-diffusion-2-1-base")

    p.add_argument("--cond-drop-prob", type=float, default=0.20)
    p.add_argument("--noise-offset", type=float, default=0.05)
    p.add_argument("--min-snr-gamma", type=float, default=5.0)
    p.add_argument("--min-timestep", type=int, default=20)
    p.add_argument("--max-timestep", type=int, default=980)

    p.add_argument("--prompt-template", type=str, default="a photo of {scene}")
    p.add_argument("--exclude-image-list", type=str, default=None)
    p.add_argument("--target-include-image-list", type=str, default=None)
    p.add_argument("--reference-include-image-list", type=str, default=None)

    p.add_argument("--min-pair-iou", type=float, default=0.18)
    p.add_argument("--max-pair-iou", type=float, default=0.62)
    p.add_argument("--min-pair-distance", type=float, default=0.20)
    p.add_argument("--max-pair-distance", type=float, default=2.2)
    p.add_argument("--min-view-cos", type=float, default=0.80)
    p.add_argument("--min-rotation-deg", type=float, default=3.0)
    p.add_argument("--max-rotation-deg", type=float, default=35.0)
    p.add_argument("--max-focal-ratio", type=float, default=1.35)
    p.add_argument("--prefilter-topk", type=int, default=64)
    p.add_argument("--targets-per-ref", type=int, default=2)
    p.add_argument("--max-pairs-per-scene", type=int, default=128)
    p.add_argument("--val-ratio", type=float, default=0.10)

    p.add_argument("--sample-every-epochs", type=int, default=1)
    p.add_argument("--sample-steps", type=int, default=40)
    p.add_argument("--sample-cfg-scale", type=float, default=4.0)
    p.add_argument("--num-debug-pairs", type=int, default=4)
    p.add_argument("--num-debug-samples", type=int, default=6)
    p.add_argument("--sample-seed-base", type=int, default=1234)

    p.add_argument("--wandb-project", type=str, default="CoupledSceneSampling")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-id", type=str, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    if args.sample_every_epochs <= 0:
        raise ValueError("--sample-every-epochs must be >= 1")
    if args.num_debug_samples <= 0:
        raise ValueError("--num-debug-samples must be >= 1")
    if args.num_debug_pairs <= 0:
        raise ValueError("--num-debug-pairs must be >= 1")
    if args.epochs <= 0:
        raise ValueError("--epochs must be >= 1")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        id=args.wandb_id,
        mode=args.wandb_mode,
        resume="allow",
        config=vars(args),
        settings=wandb.Settings(x_stats_sampling_interval=10),
    )
    if wandb.run is not None:
        wandb.run.log_code("css")

    scenes = list(dict.fromkeys((args.scenes or []) + _read_lines(args.scenes_file)))
    if len(scenes) == 0:
        raise ValueError("Provide --scenes or --scenes-file")
    print(f"Using {len(scenes)} scenes")

    exclude_image_names = load_image_name_set(args.exclude_image_list)
    target_include_image_names = load_image_name_set(args.target_include_image_list)
    reference_include_image_names = load_image_name_set(args.reference_include_image_list)

    dataset_all = SingleRefPairDataset(
        scene_dirs=scenes,
        H=args.H,
        W=args.W,
        min_pair_iou=args.min_pair_iou,
        max_pair_iou=args.max_pair_iou,
        min_pair_distance=args.min_pair_distance,
        max_pair_distance=args.max_pair_distance,
        min_view_cos=args.min_view_cos,
        min_rotation_deg=args.min_rotation_deg,
        max_rotation_deg=args.max_rotation_deg,
        max_focal_ratio=args.max_focal_ratio,
        prefilter_topk=args.prefilter_topk,
        targets_per_ref=args.targets_per_ref,
        max_pairs_per_scene=args.max_pairs_per_scene,
        exclude_image_names=exclude_image_names,
        target_include_image_names=target_include_image_names,
        reference_include_image_names=reference_include_image_names,
        prompt_template=args.prompt_template,
    )
    if len(dataset_all) < 2:
        raise ValueError("Need at least 2 pairs for train/val split.")

    train_pairs, val_pairs = split_pair_records(dataset_all.pairs, val_ratio=args.val_ratio, seed=args.seed)
    train_dataset = PairRecordViewDataset(train_pairs, H=args.H, W=args.W)
    val_dataset = PairRecordViewDataset(val_pairs, H=args.H, W=args.W)
    print(f"Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}")
    wandb.log(
        {
            "data/train_pairs": len(train_dataset),
            "data/val_pairs": len(val_dataset),
            "data/all_pairs": len(dataset_all),
        },
        step=0,
    )

    model = SingleRefPoseSD(pretrained_model=args.pretrained_model)
    model.configure_trainable(args.train_mode)
    model.configure_memory_optimizations(
        gradient_checkpointing=args.gradient_checkpointing,
        xformers_attention=args.xformers_attention,
    )

    trainable_params = model.get_trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Train mode={args.train_mode} trainable params={n_trainable:,}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": (model.device.type == "cuda"),
        "persistent_workers": (args.num_workers > 0),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_dataset, **loader_kwargs)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = _constant_with_warmup(optimizer, warmup_steps=args.warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(model.device.type == "cuda" and args.mixed_precision == "fp16"))

    global_step = 0
    try:
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch in pbar:
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(model.device, args.mixed_precision):
                    loss = model.training_step(
                        batch,
                        cond_drop_prob=args.cond_drop_prob,
                        min_timestep=args.min_timestep,
                        max_timestep=args.max_timestep,
                        noise_offset=args.noise_offset,
                        min_snr_gamma=args.min_snr_gamma,
                    )

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()

                epoch_loss += float(loss.item())
                global_step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/lr": float(lr_scheduler.get_last_lr()[0]),
                    },
                    step=global_step,
                )

            avg_loss = epoch_loss / max(1, len(train_loader))
            print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.5f}")
            wandb.log(
                {
                    "train/epoch_loss": float(avg_loss),
                    "train/epoch": int(epoch + 1),
                },
                step=global_step,
            )

            model.eval()
            log_train_sample_to_wandb(
                model,
                train_dataset,
                output_dir=output_dir,
                epoch=epoch + 1,
                global_step=global_step,
                num_steps=args.sample_steps,
                cfg_scale=args.sample_cfg_scale,
                sample_seed=args.sample_seed_base + epoch,
            )

            if (epoch + 1) % args.sample_every_epochs == 0 and len(val_dataset) > 0:
                stats = evaluate_diversity(
                    model,
                    val_dataset,
                    output_dir=output_dir / "debug_grids" / f"epoch_{epoch + 1:03d}",
                    num_pairs=args.num_debug_pairs,
                    num_samples=args.num_debug_samples,
                    sample_seed_base=args.sample_seed_base + epoch * 100_000,
                    num_steps=args.sample_steps,
                    cfg_scale=args.sample_cfg_scale,
                )
                stats_path = output_dir / "debug_grids" / f"epoch_{epoch + 1:03d}" / "metrics.json"
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
                if len(stats) > 0:
                    diversity_vals = [s["diversity_l1"] for s in stats]
                    ref_copy_vals = [s["ref_copy_l1"] for s in stats]
                    print(
                        f"  sample diversity_l1 mean={np.mean(diversity_vals):.4f} "
                        f"(saved {len(stats)} grids to {stats_path.parent})"
                    )
                    wandb.log(
                        {
                            "eval/diversity_l1_mean": float(np.mean(diversity_vals)),
                            "eval/ref_copy_l1_mean": float(np.mean(ref_copy_vals)),
                            "eval/epoch": int(epoch + 1),
                        },
                        step=global_step,
                    )

        ckpt_path = output_dir / "unet_final.pt"
        torch.save({"unet": model.unet.state_dict(), "global_step": global_step}, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        if len(val_dataset) > 0:
            model.eval()
            final_stats = evaluate_diversity(
                model,
                val_dataset,
                output_dir=output_dir / "debug_grids" / "final",
                num_pairs=args.num_debug_pairs,
                num_samples=args.num_debug_samples,
                sample_seed_base=args.sample_seed_base + 999_999,
                num_steps=args.sample_steps,
                cfg_scale=args.sample_cfg_scale,
            )
            final_stats_path = output_dir / "debug_grids" / "final" / "metrics.json"
            final_stats_path.parent.mkdir(parents=True, exist_ok=True)
            final_stats_path.write_text(json.dumps(final_stats, indent=2), encoding="utf-8")
            print(f"Saved final debug outputs: {final_stats_path.parent}")
            if len(final_stats) > 0:
                diversity_vals = [s["diversity_l1"] for s in final_stats]
                ref_copy_vals = [s["ref_copy_l1"] for s in final_stats]
                wandb.log(
                    {
                        "eval_final/diversity_l1_mean": float(np.mean(diversity_vals)),
                        "eval_final/ref_copy_l1_mean": float(np.mean(ref_copy_vals)),
                    },
                    step=global_step,
                )

        run_config_path = output_dir / "run_config.json"
        run_config_path.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(f"Saved run config: {run_config_path}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
