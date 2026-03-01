"""Generate a novel view from two arbitrary photos using DUSt3R for pose estimation.

Usage:
    python -m css.infer_photos \
        --ref1 photo1.jpg --ref2 photo2.jpg \
        --checkpoint checkpoints/unet_final.pt \
        --direction right --distance 0.3 \
        --prompt "a photo of a building"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from css.data.dataset import compute_plucker_tensor
from css.models.EMA import load_pose_sd_checkpoint
from css.models.pose_conditioned_sd import PoseConditionedSD
from css.scene_sampling import plucker_to_rgb, to_uint8

DIRECTIONS = {
    "left": np.array([-1, 0, 0], dtype=np.float64),
    "right": np.array([1, 0, 0], dtype=np.float64),
    "up": np.array([0, -1, 0], dtype=np.float64),
    "down": np.array([0, 1, 0], dtype=np.float64),
    "forward": np.array([0, 0, -1], dtype=np.float64),
    "back": np.array([0, 0, 1], dtype=np.float64),
}


def adjust_K(K: np.ndarray, src_w: int, src_h: int, W: int, H: int) -> np.ndarray:
    """Adjust intrinsics for center-crop + resize from (src_w, src_h) to (W, H)."""
    K = K.copy()
    target_aspect = W / H
    src_aspect = src_w / src_h
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


def load_and_preprocess(path: str, H: int, W: int) -> tuple[torch.Tensor, int, int]:
    """Load image, center-crop to target aspect, resize. Returns (tensor, orig_w, orig_h)."""
    with Image.open(path) as pil:
        rgb = pil.convert("RGB")
        orig_w, orig_h = rgb.size
        target_aspect = W / H
        src_aspect = orig_w / orig_h
        if src_aspect > target_aspect:
            new_w = int(orig_h * target_aspect)
            offset = (orig_w - new_w) // 2
            rgb = rgb.crop((offset, 0, offset + new_w, orig_h))
        elif src_aspect < target_aspect:
            new_h = int(orig_w / target_aspect)
            offset = (orig_h - new_h) // 2
            rgb = rgb.crop((0, offset, orig_w, offset + new_h))
        rgb = rgb.resize((W, H), Image.LANCZOS)
        arr = np.array(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1
    return tensor, orig_w, orig_h


def run_dust3r(ref1_path: str, ref2_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Run DUSt3R on two images. Returns (Ks [2,3,3], c2ws [2,4,4])."""
    dust3r_root = str(Path(__file__).resolve().parent.parent / "stable-virtual-camera" / "third_party" / "dust3r")
    if dust3r_root not in sys.path:
        sys.path.insert(0, dust3r_root)

    from seva.modules.preprocessor import Dust3rPipeline

    print("Loading DUSt3R model...")
    pipeline = Dust3rPipeline(device="cuda")
    print("Running pose estimation...")
    _imgs, Ks, c2ws, _pts, _colors = pipeline.infer_cameras_and_points(
        [ref1_path, ref2_path], niter=300,
    )
    return Ks, c2ws


def main():
    p = argparse.ArgumentParser(description="Novel view from two photos via DUSt3R + pose-conditioned SD")
    p.add_argument("--ref1", required=True, help="Path to first reference photo")
    p.add_argument("--ref2", required=True, help="Path to second reference photo")
    p.add_argument("--checkpoint", required=True, help="UNet checkpoint (.pt)")
    p.add_argument("--direction", default="right", choices=list(DIRECTIONS.keys()))
    p.add_argument("--distance", type=float, default=0.3, help="Translation magnitude")
    p.add_argument("--anchor", default="ref1", choices=["ref1", "ref2"], help="Which ref to offset from")
    p.add_argument("--prompt", default="a photo of a scene")
    p.add_argument("--output", default="output.png")
    p.add_argument("--cfg-scale", type=float, default=5.5)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--show-pluckers", action="store_true", help="Include Plucker ray direction maps in grid")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    H, W = args.H, args.W
    latent_h, latent_w = H // 8, W // 8

    # 1. Estimate poses with DUSt3R
    Ks, c2ws = run_dust3r(args.ref1, args.ref2)
    # Ks: (2, 3, 3) at original image resolution; c2ws: (2, 4, 4)
    c2w_ref1 = c2ws[0].astype(np.float64)
    c2w_ref2 = c2ws[1].astype(np.float64)
    K_ref1_orig = Ks[0].astype(np.float64)
    K_ref2_orig = Ks[1].astype(np.float64)

    baseline = np.linalg.norm(c2w_ref2[:3, 3] - c2w_ref1[:3, 3])
    print(f"DUSt3R baseline: {baseline:.4f}")

    # 2. Load images
    ref1_tensor, w1, h1 = load_and_preprocess(args.ref1, H, W)
    ref2_tensor, w2, h2 = load_and_preprocess(args.ref2, H, W)

    # 3. Adjust intrinsics for center-crop + resize
    K_ref1 = adjust_K(K_ref1_orig, w1, h1, W, H).astype(np.float32)
    K_ref2 = adjust_K(K_ref2_orig, w2, h2, W, H).astype(np.float32)

    # 4. Build target camera: same rotation as anchor, shifted position
    anchor_idx = 0 if args.anchor == "ref1" else 1
    anchor_c2w = c2ws[anchor_idx].astype(np.float64)
    local_dir = DIRECTIONS[args.direction]
    offset_world = anchor_c2w[:3, :3] @ (local_dir * args.distance)

    target_c2w = anchor_c2w.copy()
    target_c2w[:3, 3] += offset_world
    # Target uses anchor's intrinsics
    K_tgt = (K_ref1 if anchor_idx == 0 else K_ref2).copy()

    print(f"Target offset: {args.direction} {args.distance} from {args.anchor}")

    # 5. Compute ref1-anchored Plucker rays
    plucker_ref1 = compute_plucker_tensor(c2w_ref1, c2w_ref1, K_ref1, H, W, latent_h, latent_w)
    plucker_ref2 = compute_plucker_tensor(c2w_ref1, c2w_ref2, K_ref2, H, W, latent_h, latent_w)
    plucker_tgt = compute_plucker_tensor(c2w_ref1, target_c2w, K_tgt, H, W, latent_h, latent_w)

    # 6. Load model
    print("Loading model...")
    model = PoseConditionedSD()
    load_pose_sd_checkpoint(model, args.checkpoint, model.device)
    model.eval()

    # 7. Generate
    print(f"Generating ({args.num_steps} steps, cfg={args.cfg_scale})...")
    with torch.inference_mode():
        generated = model.sample(
            ref1_tensor.unsqueeze(0),
            ref2_tensor.unsqueeze(0),
            plucker_ref1.unsqueeze(0),
            plucker_ref2.unsqueeze(0),
            plucker_tgt.unsqueeze(0),
            prompt=args.prompt,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
        )

    # 8. Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save grid: [ref1 | ref2 | generated], optionally with Plucker row below
    img_row = np.concatenate([
        to_uint8(ref1_tensor),
        to_uint8(ref2_tensor),
        to_uint8(generated[0]),
    ], axis=1)
    if args.show_pluckers:
        pl_row = np.concatenate([
            plucker_to_rgb(plucker_ref1, H, W),
            plucker_to_rgb(plucker_ref2, H, W),
            plucker_to_rgb(plucker_tgt, H, W),
        ], axis=1)
        grid = np.concatenate([img_row, pl_row], axis=0)
    else:
        grid = img_row
    Image.fromarray(grid).save(out_path)
    print(f"Saved: {out_path}  [Ref1 | Ref2 | Generated]")


if __name__ == "__main__":
    main()
