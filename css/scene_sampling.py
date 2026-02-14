"""Utilities for split-aware sampling from reconstructed scenes."""

from pathlib import Path

import numpy as np
import torch

from css.data.colmap_reader import Camera, ImageData, read_scene
from css.data.dataset import build_scaled_intrinsics, compute_plucker_tensor, load_image_tensor


def load_scene_pools(
    scene_dir: Path,
    exclude_image_names: set[str] | None = None,
    target_include_image_names: set[str] | None = None,
    reference_include_image_names: set[str] | None = None,
) -> tuple[dict[int, Camera], Path, list[ImageData], list[ImageData]]:
    cameras, images = read_scene(scene_dir)
    images_dir = scene_dir / "images"

    valid_images = [img for img in images.values() if (images_dir / img.name).exists()]
    valid_images.sort(key=lambda x: x.id)
    if exclude_image_names is not None:
        valid_images = [img for img in valid_images if img.name not in exclude_image_names]

    target_images = valid_images
    if target_include_image_names is not None:
        target_images = [img for img in target_images if img.name in target_include_image_names]

    reference_images = valid_images
    if reference_include_image_names is not None:
        reference_images = [img for img in reference_images if img.name in reference_include_image_names]

    return cameras, images_dir, target_images, reference_images


def compute_viewing_direction(c2w: np.ndarray) -> np.ndarray:
    return -c2w[:3, 2]


def find_best_references(
    target_img: ImageData,
    all_images: list[ImageData],
    max_dist: float = 2.0,
    min_dir_sim: float = 0.3,
    min_ref_spacing: float = 0.3,
) -> tuple[ImageData, ImageData]:
    target_pos = target_img.c2w[:3, 3]
    target_dir = compute_viewing_direction(target_img.c2w)

    candidates = []
    for img in all_images:
        if img.id == target_img.id:
            continue

        pos = img.c2w[:3, 3]
        direction = compute_viewing_direction(img.c2w)

        distance = np.linalg.norm(pos - target_pos)
        if distance > max_dist:
            continue

        dir_sim = float(np.dot(target_dir, direction))
        if dir_sim < min_dir_sim:
            continue

        score = distance * (2.0 - dir_sim)
        candidates.append((score, img))

    if len(candidates) < 2:
        raise ValueError(f"Not enough valid references found for target {target_img.name}")

    candidates.sort(key=lambda x: x[0])
    ref1 = candidates[0][1]

    for _, img in candidates[1:]:
        spacing = np.linalg.norm(img.c2w[:3, 3] - ref1.c2w[:3, 3])
        if spacing >= min_ref_spacing:
            return ref1, img

    return ref1, candidates[1][1]


def build_single_sample(
    cameras: dict[int, Camera],
    images_dir: Path,
    ref1_img: ImageData,
    ref2_img: ImageData,
    target_img: ImageData,
    H: int,
    W: int,
) -> dict[str, torch.Tensor]:
    latent_h, latent_w = H // 8, W // 8

    ref1_tensor = load_image_tensor(images_dir, ref1_img.name, H, W)
    ref2_tensor = load_image_tensor(images_dir, ref2_img.name, H, W)
    target_tensor = load_image_tensor(images_dir, target_img.name, H, W)

    K_ref1 = build_scaled_intrinsics(cameras[ref1_img.camera_id], H, W)
    K_ref2 = build_scaled_intrinsics(cameras[ref2_img.camera_id], H, W)
    K_tgt = build_scaled_intrinsics(cameras[target_img.camera_id], H, W)

    plucker_ref1 = compute_plucker_tensor(ref1_img.c2w, ref1_img.c2w, K_ref1, latent_h, latent_w)
    plucker_ref2 = compute_plucker_tensor(ref1_img.c2w, ref2_img.c2w, K_ref2, latent_h, latent_w)
    plucker_target = compute_plucker_tensor(ref1_img.c2w, target_img.c2w, K_tgt, latent_h, latent_w)

    return {
        "ref1_img": ref1_tensor.unsqueeze(0),
        "ref2_img": ref2_tensor.unsqueeze(0),
        "target_img": target_tensor.unsqueeze(0),
        "plucker_ref1": plucker_ref1.unsqueeze(0),
        "plucker_ref2": plucker_ref2.unsqueeze(0),
        "plucker_target": plucker_target.unsqueeze(0),
    }


def to_uint8(t: torch.Tensor) -> np.ndarray:
    return ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()


def build_comparison_grid(
    ref1_img: torch.Tensor,
    ref2_img: torch.Tensor,
    target_img: torch.Tensor,
    generated_img: torch.Tensor,
) -> np.ndarray:
    return np.concatenate([
        to_uint8(ref1_img),
        to_uint8(ref2_img),
        to_uint8(target_img),
        to_uint8(generated_img),
    ], axis=1)
