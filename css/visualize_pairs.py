"""Visualize ref/target pairs for a scene, optionally with model generations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from css.data.colmap_reader import read_scene
from css.data.dataset import (
    build_cropped_scaled_intrinsics,
    clean_scene_prompt_name,
    load_image_tensor,
    read_scene_prompt_name,
)
from css.data.iou import (
    build_camera_frustum_geometry,
    compute_covisibility,
    compute_frustum_iou_from_geometries,
    compute_reference_depth,
)


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    return ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()


def build_pairs(scene_dir: Path, H: int, W: int,
                min_covis: float, max_covis: float,
                max_pairs: int) -> list[dict]:
    cameras, images = read_scene(scene_dir)
    images_dir = scene_dir / "images"

    disk_files: set[str] = set()
    for p in images_dir.rglob("*"):
        if p.is_file():
            disk_files.add(p.relative_to(images_dir).as_posix())

    valid = []
    resolved: dict[int, str] = {}
    for img in images.values():
        name = img.name.replace("\\", "/")
        if name in disk_files:
            resolved[img.id] = name
            valid.append(img)

    if len(valid) < 2:
        return []

    positions = {img.id: img.c2w[:3, 3].astype(np.float64) for img in valid}
    # K_by_id = {img.id: build_cropped_scaled_intrinsics(cameras[img.camera_id], H, W) for img in valid}
    # d_ref = compute_reference_depth(positions)
    # geom = {
    #     img.id: build_camera_frustum_geometry(img.c2w, K_by_id[img.id], H, W, d_ref)
    #     for img in valid
    # }

    pairs: list[dict] = []
    for i, ref in enumerate(valid):
        for tgt in valid[i + 1:]:
            covis = compute_covisibility(ref, tgt)
            if covis < min_covis or covis > max_covis:
                continue

            dist = float(np.linalg.norm(positions[ref.id] - positions[tgt.id]))
            key = (min(ref.id, tgt.id), max(ref.id, tgt.id))
            # fiou = compute_frustum_iou_from_geometries(geom[key[0]], geom[key[1]])

            pairs.append({
                "images_dir": images_dir,
                "ref_name": resolved[ref.id],
                "target_name": resolved[tgt.id],
                "covisibility": covis,
                # "frustum_iou": fiou,
                "distance": dist,
            })

    if len(pairs) > max_pairs:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in indices]

    pairs.sort(key=lambda p: p["covisibility"], reverse=True)
    return pairs


def make_vis(ref_img: np.ndarray, tgt_img: np.ndarray,
             covis: float, dist: float,
             gen_img: np.ndarray | None = None) -> Image.Image:
    h, w = ref_img.shape[:2]
    label_h = 32
    n_panels = 3 if gen_img is not None else 2
    canvas_w = w * n_panels
    canvas_h = h + label_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(Image.fromarray(ref_img), (0, label_h))
    canvas.paste(Image.fromarray(tgt_img), (w, label_h))
    if gen_img is not None:
        canvas.paste(Image.fromarray(gen_img), (w * 2, label_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    draw.text((4, 4), "Reference", fill=(0, 0, 0), font=font)
    draw.text((w + 4, 4), f"Target  |  covis={covis:.3f}  dist={dist:.2f}",
              fill=(0, 0, 0), font=font)
    if gen_img is not None:
        draw.text((w * 2 + 4, 4), "Generated", fill=(0, 0, 0), font=font)

    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize ref/target pairs for a scene")
    p.add_argument("scene", type=str, help="Path to scene directory")
    p.add_argument("--output", type=str, default=None,
                   help="Output folder (default: ./vis_pairs/<scene_name>)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to trained SingleRefSD checkpoint for generation")
    p.add_argument("--prompt", type=str, default=None)

    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--min-covisibility", type=float, default=0.23)
    p.add_argument("--max-covisibility", type=float, default=0.58)
    p.add_argument("--max-pairs", type=int, default=128)

    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=3)
    p.add_argument("--seed", type=int, default=15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene)
    scene_name = scene_dir.name

    output_dir = Path(args.output) if args.output else Path("vis_pairs") / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(scene_dir, args.H, args.W,
                        args.min_covisibility, args.max_covisibility,
                        args.max_pairs)
    print(f"{scene_name}: {len(pairs)} pairs")
    if not pairs:
        return

    prompt = args.prompt or f"a photo of {clean_scene_prompt_name(read_scene_prompt_name(scene_dir))}"

    model = None
    if args.checkpoint:
        from css.debug_single_ref_experiment import SingleRefSD
        model = SingleRefSD()
        ckpt = torch.load(args.checkpoint, map_location=model.device, weights_only=True)
        model.unet.load_state_dict(ckpt["unet"])
        model.eval()
        print(f"Loaded checkpoint: {args.checkpoint}")

    for i, pair in enumerate(pairs):
        ref_img, _, _ = load_image_tensor(pair["images_dir"], pair["ref_name"], args.H, args.W)
        tgt_img, _, _ = load_image_tensor(pair["images_dir"], pair["target_name"], args.H, args.W)

        gen_np = None
        if model is not None:
            with torch.inference_mode():
                gen = model.sample(ref_img=ref_img.unsqueeze(0), prompt=prompt,
                                   num_steps=args.sample_steps, cfg_scale=args.cfg_scale,
                                   seed=args.seed + i)[0].cpu()
            gen_np = _to_uint8(gen)

        vis = make_vis(_to_uint8(ref_img), _to_uint8(tgt_img),
                       pair["covisibility"], pair["distance"], gen_np)

        ref_short = Path(pair["ref_name"]).stem
        tgt_short = Path(pair["target_name"]).stem
        vis.save(output_dir / f"{i:03d}_cv{pair['covisibility']:.3f}_d{pair['distance']:.2f}_{ref_short}_{tgt_short}.png")

    print(f"Saved {len(pairs)} images to {output_dir}")


if __name__ == "__main__":
    main()
