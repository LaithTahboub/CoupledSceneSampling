"""Triplet inference: generate a target view from two reference views.

Two camera modes:
  --scene  : use COLMAP cameras from a reconstructed scene directory
  (default): use MAST3R to estimate camera poses from arbitrary photos
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from css.data.dataset import (
    build_cropped_scaled_intrinsics,
    compute_plucker_tensor,
)
from css.inference.scene_sampling import build_comparison_grid, load_scene_pools
from css.models.EMA import load_pose_sd_checkpoint

_SEVA_ROOT = str(Path(__file__).resolve().parent.parent.parent / "stable-virtual-camera")
if _SEVA_ROOT not in sys.path:
    sys.path.insert(0, _SEVA_ROOT)

def load_image_tensor_exact(path: Path, H: int, W: int, *, pad: bool = False) -> torch.Tensor:
    """Load image, resize to exactly (W, H), return tensor (1, 3, H, W) in [-1, 1].

    When pad=False (default), center-crops to match the target aspect ratio.
    When pad=True, fits the image inside (W, H) with black letterboxing.
    """
    img = Image.open(path).convert("RGB")
    src_w, src_h = img.size
    target_aspect = W / H
    src_aspect = src_w / src_h

    if pad:
        # Fit inside (W, H) preserving aspect ratio, pad the rest with black
        if src_aspect > target_aspect:
            # Image is wider: fit to width, pad height
            new_w = W
            new_h = int(round(W / src_aspect))
        else:
            # Image is taller: fit to height, pad width
            new_h = H
            new_w = int(round(H * src_aspect))
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        paste_x = (W - new_w) // 2
        paste_y = (H - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        img = canvas
    else:
        if src_aspect > target_aspect:
            new_w = int(round(src_h * target_aspect))
            left = (src_w - new_w) // 2
            img = img.crop((left, 0, left + new_w, src_h))
        elif src_aspect < target_aspect:
            new_h = int(round(src_w / target_aspect))
            top = (src_h - new_h) // 2
            img = img.crop((0, top, src_w, top + new_h))
        img = img.resize((W, H), Image.Resampling.BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def adjust_K_for_crop_resize(K: np.ndarray, src_w: int, src_h: int, H: int, W: int,
                              *, pad: bool = False) -> np.ndarray:
    """Adjust intrinsics for center-crop + resize (or pad + resize) from (src_w, src_h) to (W, H)."""
    K = K.copy()
    target_aspect = W / H
    src_aspect = src_w / src_h

    if pad:
        # Fit-inside: scale preserving aspect, then offset for centering
        if src_aspect > target_aspect:
            scale = W / src_w
            new_h = int(round(W / src_aspect))
            paste_y = (H - new_h) // 2
            K[0] *= scale
            K[1] *= scale
            K[1, 2] += paste_y
        else:
            scale = H / src_h
            new_w = int(round(H * src_aspect))
            paste_x = (W - new_w) // 2
            K[0] *= scale
            K[1] *= scale
            K[0, 2] += paste_x
    else:
        if src_aspect > target_aspect:
            new_w = int(src_h * target_aspect)
            K[0, 2] -= (src_w - new_w) / 2.0
            K[0] *= W / new_w
            K[1] *= H / src_h
        elif src_aspect < target_aspect:
            new_h = int(src_w / target_aspect)
            K[1, 2] -= (src_h - new_h) / 2.0
            K[0] *= W / src_w
            K[1] *= H / new_h
        else:
            K[0] *= W / src_w
            K[1] *= H / src_h

    return K


def main():
    parser = argparse.ArgumentParser(description="Triplet inference with MAST3R + PoseSD")
    parser.add_argument("--ref1", required=True, help="Reference image 1 (path or name within scene)")
    parser.add_argument("--ref2", required=True, help="Reference image 2 (path or name within scene)")
    parser.add_argument("--target", required=True, help="Target image (path or name within scene)")
    parser.add_argument("--checkpoint", required=True, help="PoseSD checkpoint")
    parser.add_argument("--scene", default=None, help="Scene directory with COLMAP data (skip MAST3R)")
    parser.add_argument("--output", default="output.png")
    parser.add_argument("--arch", choices=["NEW", "OLD"], default="NEW")
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="Geometry CFG scale")
    parser.add_argument("--cfg-text", type=float, default=4.5, help="Text CFG scale (0=ignore text)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples; increments seed each time")
    parser.add_argument("--prompt", default="", help="Text prompt for generation")
    parser.add_argument("--no-crop", action="store_true",
                        help="Pad (letterbox) instead of center-crop to preserve full FoV")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    H, W = args.H, args.W
    lh, lw = H // 8, W // 8

    if args.scene:
        # --- COLMAP cameras from scene directory ---
        print(f"Loading COLMAP scene: {args.scene}")
        scene_dir = Path(args.scene)
        cameras, images_dir, all_images, _ = load_scene_pools(scene_dir)

        # Match ref1, ref2, target by relative path within images/, basename, or absolute path
        def find_image(query, all_imgs):
            query_path = Path(query)
            query_name = query_path.name
            # Try as path relative to images_dir (handles absolute paths pointing into the scene)
            try:
                query_rel = query_path.resolve().relative_to(images_dir.resolve()).as_posix()
            except ValueError:
                query_rel = None
            for img in all_imgs:
                if img.name == query or Path(img.name).name == query_name:
                    return img
                if query_rel and img.name == query_rel:
                    return img
            raise ValueError(f"Image '{query}' not found in scene. "
                             f"Available: {[img.name for img in all_imgs[:10]]}...")

        ref1_data = find_image(args.ref1, all_images)
        ref2_data = find_image(args.ref2, all_images)
        tgt_data = find_image(args.target, all_images)

        K_ref1 = build_cropped_scaled_intrinsics(cameras[ref1_data.camera_id], H, W)
        K_ref2 = build_cropped_scaled_intrinsics(cameras[ref2_data.camera_id], H, W)
        K_tgt = build_cropped_scaled_intrinsics(cameras[tgt_data.camera_id], H, W)

        pl_ref1 = compute_plucker_tensor(ref1_data.c2w, ref1_data.c2w, K_ref1, H, W, lh, lw).unsqueeze(0)
        pl_ref2 = compute_plucker_tensor(ref1_data.c2w, ref2_data.c2w, K_ref2, H, W, lh, lw).unsqueeze(0)
        pl_tgt = compute_plucker_tensor(ref1_data.c2w, tgt_data.c2w, K_tgt, H, W, lh, lw).unsqueeze(0)

        from css.data.dataset import load_image_tensor
        ref1_tensor, _, _ = load_image_tensor(images_dir, ref1_data.name, H, W)
        ref2_tensor, _, _ = load_image_tensor(images_dir, ref2_data.name, H, W)
        ref1_tensor = ref1_tensor.unsqueeze(0)
        ref2_tensor = ref2_tensor.unsqueeze(0)
    else:
        # --- Estimate poses with MAST3R ---
        from seva.modules.preprocessor import get_pose_pipeline  # noqa: E402
        print("Running MAST3R...")
        pipe = get_pose_pipeline("mast3r", device=args.device)
        imgs_np, Ks, c2ws, _, _ = pipe.infer_cameras_and_points(
            [args.ref1, args.ref2, args.target]
        )
        orig_sizes = [(img.shape[1], img.shape[0]) for img in imgs_np]  # (w, h)

        del pipe
        torch.cuda.empty_cache()

        # --- Adjust intrinsics for center-crop (or pad) + resize to (H, W) ---
        K_adj = [
            adjust_K_for_crop_resize(Ks[i], orig_sizes[i][0], orig_sizes[i][1], H, W,
                                     pad=args.no_crop)
            for i in range(3)
        ]

        # --- Compute pluckers (anchored to ref1) ---
        c2w_ref1, c2w_ref2, c2w_tgt = c2ws[0], c2ws[1], c2ws[2]

        pl_ref1 = compute_plucker_tensor(c2w_ref1, c2w_ref1, K_adj[0], H, W, lh, lw).unsqueeze(0)
        pl_ref2 = compute_plucker_tensor(c2w_ref1, c2w_ref2, K_adj[1], H, W, lh, lw).unsqueeze(0)
        pl_tgt = compute_plucker_tensor(c2w_ref1, c2w_tgt, K_adj[2], H, W, lh, lw).unsqueeze(0)

        # --- Load images ---
        ref1_tensor = load_image_tensor_exact(Path(args.ref1), H, W, pad=args.no_crop)
        ref2_tensor = load_image_tensor_exact(Path(args.ref2), H, W, pad=args.no_crop)

    assert ref1_tensor.shape[-2:] == (H, W)
    assert ref2_tensor.shape[-2:] == (H, W)
    # --- Load PoseSD ---
    print("Loading PoseSD...")
    if args.arch == "OLD":
        from css.old.pose_sd import PoseSD
    else:
        from css.models.pose_sd import PoseSD

    model = PoseSD(device=args.device)
    load_pose_sd_checkpoint(model, args.checkpoint, model.device)
    model.eval()

    # --- Generate ---
    if args.scene:
        target_tensor, _, _ = load_image_tensor(images_dir, tgt_data.name, H, W)
        target_tensor = target_tensor.unsqueeze(0)
    else:
        target_tensor = load_image_tensor_exact(Path(args.target), H, W, pad=args.no_crop)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed + i for i in range(args.num_samples)]
    print(f"Generating {args.num_samples} sample(s) (seeds={seeds})...")
    with torch.inference_mode():
        generated = model.sample(
            ref1_img=ref1_tensor,
            ref2_img=ref2_tensor,
            pl_ref1=pl_ref1,
            pl_ref2=pl_ref2,
            pl_tgt=pl_tgt,
            prompt=args.prompt,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            cfg_text=args.cfg_text,
            seed=seeds if args.num_samples > 1 else args.seed,
        )

    for i, current_seed in enumerate(seeds):
        grid = build_comparison_grid(
            ref1_tensor[0], ref2_tensor[0],
            target_tensor[0], generated[i],
            prompt=args.prompt if args.prompt else None,
        )

        if args.num_samples == 1:
            save_path = output_path
        else:
            save_path = output_path.with_stem(f"{output_path.stem}_seed{current_seed}")
        Image.fromarray(grid).save(save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()