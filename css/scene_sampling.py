"""Utilities for split-aware sampling from reconstructed scenes."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from css.data.colmap_reader import Camera, ImageData, read_scene
from css.data.dataset import (
    build_cropped_scaled_intrinsics,
    compute_frustum_iou,
    compute_plucker_tensor,
    compute_reference_depth,
    load_image_tensor,
    name_allowed,
)


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


def _resolve_image_name(
    image_name: str,
    rel_set: set[str],
    unique_basename: dict[str, str],
) -> str | None:
    norm = image_name.replace("\\", "/")
    if norm in rel_set:
        return norm
    return unique_basename.get(Path(norm).name)


def _passes_filter(
    image_names: tuple[str, str],
    scene_name: str,
    name_filter: set[str] | None,
) -> bool:
    if name_filter is None:
        return True
    return any(name_allowed(name_filter, scene_name, n) for n in image_names)


def load_scene_pools(
    scene_dir: Path,
    exclude_image_names: set[str] | None = None,
    target_include_image_names: set[str] | None = None,
    reference_include_image_names: set[str] | None = None,
) -> tuple[dict[int, Camera], Path, list[ImageData], list[ImageData]]:
    cameras, images = read_scene(scene_dir)
    images_dir = scene_dir / "images"

    rel_set, unique_basename = _index_scene_images(images_dir)
    valid_images: list[ImageData] = []
    raw_name_by_id: dict[int, str] = {}
    for img in images.values():
        resolved = _resolve_image_name(img.name, rel_set, unique_basename)
        if resolved is None:
            continue
        raw_name_by_id[img.id] = img.name
        valid_images.append(
            ImageData(
                id=img.id,
                camera_id=img.camera_id,
                name=resolved,
                c2w=img.c2w,
            )
        )
    valid_images.sort(key=lambda x: x.id)

    if exclude_image_names is not None:
        valid_images = [
            img
            for img in valid_images
            if not _passes_filter(
                (img.name, raw_name_by_id.get(img.id, img.name)),
                scene_dir.name,
                exclude_image_names,
            )
        ]

    target_images = valid_images
    if target_include_image_names is not None:
        target_images = [
            img
            for img in target_images
            if _passes_filter(
                (img.name, raw_name_by_id.get(img.id, img.name)),
                scene_dir.name,
                target_include_image_names,
            )
        ]

    reference_images = valid_images
    if reference_include_image_names is not None:
        reference_images = [
            img
            for img in reference_images
            if _passes_filter(
                (img.name, raw_name_by_id.get(img.id, img.name)),
                scene_dir.name,
                reference_include_image_names,
            )
        ]

    return cameras, images_dir, target_images, reference_images


def find_best_references(
    target_img: ImageData,
    all_images: list[ImageData],
    cameras: dict[int, Camera],
    H: int = 512,
    W: int = 512,
    max_dist: float = 2.5,
    min_pair_iou: float = 0.15,
    min_ref_spacing: float = 0.3,
) -> tuple[ImageData, ImageData]:
    target_pos = target_img.c2w[:3, 3]
    K_tgt = build_cropped_scaled_intrinsics(cameras[target_img.camera_id], H, W)

    positions = {img.id: img.c2w[:3, 3] for img in all_images}
    positions[target_img.id] = target_pos
    d_ref = compute_reference_depth(positions)

    candidates = []
    for img in all_images:
        if img.id == target_img.id:
            continue

        distance = float(np.linalg.norm(img.c2w[:3, 3] - target_pos))
        if distance > max_dist:
            continue

        K_ref = build_cropped_scaled_intrinsics(cameras[img.camera_id], H, W)
        iou = compute_frustum_iou(
            target_img.c2w, K_tgt,
            img.c2w, K_ref,
            H, W, d_ref,
        )
        if iou < min_pair_iou:
            continue

        candidates.append((iou, img))

    if len(candidates) < 2:
        raise ValueError(f"Not enough valid references found for target {target_img.name}")

    # Sort by IoU descending (best overlap first).
    candidates.sort(key=lambda x: x[0], reverse=True)
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

    ref1_tensor, _, _ = load_image_tensor(images_dir, ref1_img.name, H, W)
    ref2_tensor, _, _ = load_image_tensor(images_dir, ref2_img.name, H, W)
    target_tensor, _, _ = load_image_tensor(images_dir, target_img.name, H, W)

    K_ref1 = build_cropped_scaled_intrinsics(cameras[ref1_img.camera_id], H, W)
    K_ref2 = build_cropped_scaled_intrinsics(cameras[ref2_img.camera_id], H, W)
    K_tgt = build_cropped_scaled_intrinsics(cameras[target_img.camera_id], H, W)

    # ref1-anchored pluckers for cat3d-style channel concat.
    plucker_ref1 = compute_plucker_tensor(ref1_img.c2w, ref1_img.c2w, K_ref1, H, W, latent_h, latent_w)
    plucker_ref2 = compute_plucker_tensor(ref1_img.c2w, ref2_img.c2w, K_ref2, H, W, latent_h, latent_w)
    plucker_tgt = compute_plucker_tensor(ref1_img.c2w, target_img.c2w, K_tgt, H, W, latent_h, latent_w)

    return {
        "ref1_img": ref1_tensor.unsqueeze(0),
        "ref2_img": ref2_tensor.unsqueeze(0),
        "target_img": target_tensor.unsqueeze(0),
        "plucker_ref1": plucker_ref1.unsqueeze(0),
        "plucker_ref2": plucker_ref2.unsqueeze(0),
        "plucker_tgt": plucker_tgt.unsqueeze(0),
    }


def to_uint8(t: torch.Tensor) -> np.ndarray:
    return ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()


def plucker_to_rgb(plucker: torch.Tensor, H: int, W: int) -> np.ndarray:
    """Visualize Plucker ray map as RGB using direction channels.

    The first 3 channels are the normalized ray direction (d_x, d_y, d_z),
    mapped from [-1, 1] to [0, 255] and upsampled to (H, W).

    Args:
        plucker: (6, h, w) Plucker ray tensor at latent resolution.
        H, W: target output resolution.

    Returns:
        (H, W, 3) uint8 numpy array.
    """
    dirs = plucker[:3].float().cpu()  # (3, h, w)
    dirs = torch.nn.functional.interpolate(
        dirs.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False,
    )[0]  # (3, H, W)
    return ((dirs.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).numpy()


def build_comparison_grid(
    ref1_img: torch.Tensor,
    ref2_img: torch.Tensor,
    target_img: torch.Tensor,
    generated_img: torch.Tensor,
    prompt: str | None = None,
    pluckers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> np.ndarray:
    """Build a horizontal comparison grid.

    Args:
        ref1_img, ref2_img, target_img, generated_img: (3, H, W) image tensors.
        prompt: optional text overlay.
        pluckers: optional (pl_ref1, pl_ref2, pl_tgt) each (6, h, w).
            When provided, a second row of Plucker direction visualizations
            is appended below the images.  The fourth column (generated) gets
            a blank placeholder since there is no Plucker map for it.
    """
    H = ref1_img.shape[1]
    W = ref1_img.shape[2]

    img_row = np.concatenate([
        to_uint8(ref1_img),
        to_uint8(ref2_img),
        to_uint8(target_img),
        to_uint8(generated_img),
    ], axis=1)

    if pluckers is not None:
        pl_ref1, pl_ref2, pl_tgt = pluckers
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        pl_row = np.concatenate([
            plucker_to_rgb(pl_ref1, H, W),
            plucker_to_rgb(pl_ref2, H, W),
            plucker_to_rgb(pl_tgt, H, W),
            blank,
        ], axis=1)
        grid = np.concatenate([img_row, pl_row], axis=0)
    else:
        grid = img_row

    if not prompt:
        return grid

    prompt_text = "prompt: " + " ".join(prompt.strip().split())
    if len(prompt_text) > 220:
        prompt_text = prompt_text[:217] + "..."
    title_h = 28
    canvas = Image.new("RGB", (grid.shape[1], grid.shape[0] + title_h), color=(255, 255, 255))
    canvas.paste(Image.fromarray(grid), (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), prompt_text, fill=(0, 0, 0))
    return np.array(canvas)
