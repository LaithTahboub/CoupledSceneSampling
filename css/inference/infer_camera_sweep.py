"""Generate images on MEDIUM-difficulty triplets from COLMAP scenes.

Two modes:
  eval  – generate at the target pose (identical to eval_medium.sh)
  sweep – for each triplet, offset the target camera and generate a sweep grid

Both modes auto-select MEDIUM-difficulty triplets. Use --scene to override
to a single scene, otherwise iterates over test scenes from the split.

Usage:
    # Eval mode — identical to eval_medium.sh:
    python -m css.inference.infer_camera_sweep \
        --checkpoint checkpoints/pose_sd_v5/unet_step_184000.pt

    # Specific scene:
    python -m css.inference.infer_camera_sweep \
        --checkpoint checkpoints/pose_sd_v5/unet_step_184000.pt \
        --scene MegaScenes/025_796__Duomo__Milan

    # Sweep mode — camera offset sweep per triplet:
    python -m css.inference.infer_camera_sweep \
        --checkpoint checkpoints/pose_sd_v5/unet_step_184000.pt \
        --mode sweep --direction right --offsets -0.5 0.0 0.5 1.0
"""

import argparse
import json
import random
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from css.data.dataset import (
    build_cropped_scaled_intrinsics,
    compute_plucker_tensor,
)
from css.data.iou import compute_covisibility
from css.models.EMA import load_pose_sd_checkpoint


# MEDIUM difficulty thresholds (same as eval_medium.sh)
MEDIUM_MIN_COVIS = 0.25
MEDIUM_MAX_COVIS = 0.50
MEDIUM_MIN_DIST = 0.08
MEDIUM_MAX_DIST = 0.40

DIRECTIONS = {
    "right": np.array([1, 0, 0], dtype=np.float64),
    "left": np.array([-1, 0, 0], dtype=np.float64),
    "up": np.array([0, -1, 0], dtype=np.float64),
    "down": np.array([0, 1, 0], dtype=np.float64),
    "forward": np.array([0, 0, -1], dtype=np.float64),
    "back": np.array([0, 0, 1], dtype=np.float64),
}

_DEFAULT_SPLIT = "splits/pose_sd_seed42"
_DEFAULT_DATA_ROOT = "/fs/nexus-scratch/ltahboub/MegaScenes"


def find_medium_triplets(target_images, reference_images, max_triplets=3):
    """Find MEDIUM-difficulty (ref1, ref2, target) triplets — same logic as eval_medium.sh."""
    triplets = []
    shuffled_targets = list(target_images)
    random.shuffle(shuffled_targets)

    for target_img in shuffled_targets:
        if len(triplets) >= max_triplets:
            break

        target_pos = target_img.c2w[:3, 3].astype(np.float64)
        candidates = []
        for ref in reference_images:
            if ref.id == target_img.id:
                continue
            dist = float(np.linalg.norm(ref.c2w[:3, 3].astype(np.float64) - target_pos))
            if dist < MEDIUM_MIN_DIST or dist > MEDIUM_MAX_DIST:
                continue
            covis = float(compute_covisibility(target_img, ref))
            if covis < MEDIUM_MIN_COVIS or covis > MEDIUM_MAX_COVIS:
                continue
            candidates.append((covis, dist, ref))

        if len(candidates) < 2:
            continue

        # ref1 = highest covisibility, ref2 = next best not too close to ref1
        candidates.sort(key=lambda x: x[0], reverse=True)
        ref1 = candidates[0][2]
        ref2 = None
        for _, _, r in candidates[1:]:
            ref_dist = float(np.linalg.norm(r.c2w[:3, 3] - ref1.c2w[:3, 3]))
            if ref_dist >= MEDIUM_MIN_DIST:
                ref2 = r
                break
        if ref2 is None:
            ref2 = candidates[1][2]

        triplets.append((ref1, ref2, target_img))

    return triplets


def offset_c2w(c2w: np.ndarray, local_dir: np.ndarray, amount: float) -> np.ndarray:
    out = c2w.copy()
    out[:3, 3] += c2w[:3, :3] @ (local_dir * amount)
    return out


def generate_eval(model, sample, args, to_uint8, build_comparison_grid, caption=""):
    """Eval mode: generate at the target pose with multiple seeds."""
    images = []
    for si in range(args.samples_per_triplet):
        s = args.seed + si
        with torch.inference_mode():
            generated = model.sample(
                ref1_img=sample["ref1_img"],
                ref2_img=sample["ref2_img"],
                pl_ref1=sample["plucker_ref1"],
                pl_ref2=sample["plucker_ref2"],
                pl_tgt=sample["plucker_tgt"],
                prompt=caption,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                cfg_text=args.cfg_text,
                seed=s,
            )
        grid = build_comparison_grid(
            sample["ref1_img"][0], sample["ref2_img"][0],
            sample["target_img"][0], generated[0],
            prompt=caption,
        )
        images.append((f"seed{s}", grid))
    return images


def generate_sweep(model, sample, ref1, tgt, cameras, args,
                   to_uint8, build_cropped_scaled_intrinsics_fn, H, W, lh, lw, caption=""):
    """Sweep mode: offset target camera, generate one grid per seed."""
    baseline = float(np.linalg.norm(ref1.c2w[:3, 3] - tgt.c2w[:3, 3]))
    local_dir = DIRECTIONS[args.direction]
    K_tgt = build_cropped_scaled_intrinsics_fn(cameras[tgt.camera_id], H, W)

    top_row = np.concatenate([
        to_uint8(sample["ref1_img"][0]),
        to_uint8(sample["ref2_img"][0]),
        to_uint8(sample["target_img"][0]),
    ], axis=1)

    results = []
    for si in range(args.samples_per_triplet):
        s = args.seed + si
        sweep_imgs = []
        for offset in args.offsets:
            delta = offset * baseline
            c2w_offset = offset_c2w(tgt.c2w, local_dir, delta)
            pl_tgt = compute_plucker_tensor(ref1.c2w, c2w_offset, K_tgt, H, W, lh, lw).unsqueeze(0)

            with torch.inference_mode():
                generated = model.sample(
                    ref1_img=sample["ref1_img"],
                    ref2_img=sample["ref2_img"],
                    pl_ref1=sample["plucker_ref1"],
                    pl_ref2=sample["plucker_ref2"],
                    pl_tgt=pl_tgt,
                    prompt=caption,
                    num_steps=args.num_steps,
                    cfg_scale=args.cfg_scale,
                    cfg_text=args.cfg_text,
                    seed=s,
                )
            sweep_imgs.append(to_uint8(generated[0]))

        # Build grid: top = [ref1 | ref2 | GT], bottom = [sweep outputs]
        bottom_row = np.concatenate(sweep_imgs, axis=1)
        target_w = max(top_row.shape[1], bottom_row.shape[1])
        def pad_to(arr, w):
            if arr.shape[1] < w:
                return np.concatenate([arr, np.zeros((H, w - arr.shape[1], 3), dtype=np.uint8)], axis=1)
            return arr

        grid = np.concatenate([pad_to(top_row, target_w), pad_to(bottom_row, target_w)], axis=0)

        # Add caption text below
        if caption:
            text = caption.strip()
            if len(text) > 200:
                text = text[:197] + "..."
            text_h = 24
            canvas = Image.new("RGB", (grid.shape[1], grid.shape[0] + text_h), (255, 255, 255))
            canvas.paste(Image.fromarray(grid), (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((4, grid.shape[0] + 4), text, fill=(0, 0, 0))
            grid = np.array(canvas)

        results.append((f"sweep_{args.direction}_seed{s}", grid))

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/vulcanscratch/ltahboub/CoupledSceneSampling/checkpoints/pose_sd_v4/unet_step_160000.pt")
    p.add_argument("--arch", choices=["OLD", "NEW"], default="OLD",
                    help="OLD = css.old.pose_sd, NEW = css.models.pose_sd")
    p.add_argument("--mode", choices=["eval", "sweep"], default="sweep",
                    help="eval = generate at target pose (like eval_medium.sh), "
                         "sweep = offset target camera per triplet")
    # Scene selection: --scene overrides to one scene, otherwise use split
    p.add_argument("--scene", default=None,
                    help="Single scene path (overrides --num-scenes)")
    p.add_argument("--split-dir", default=_DEFAULT_SPLIT)
    p.add_argument("--data-root", default=_DEFAULT_DATA_ROOT)
    p.add_argument("--num-scenes", type=int, default=5)
    p.add_argument("--scene-idx", type=int, default=0,
                    help="Start index into test_scenes list")
    # Triplet selection
    p.add_argument("--targets-per-scene", type=int, default=3)
    p.add_argument("--samples-per-triplet", type=int, default=3,
                    help="Seeds per triplet (eval mode only)")
    # Sweep params
    p.add_argument("--direction", default="right", choices=list(DIRECTIONS.keys()))
    p.add_argument("--offsets", type=float, nargs="+", default=[-0.5, 0.0, 0.5, 1.0])
    # Captions
    p.add_argument("--caption-dir", default=None,
                    help="Directory with per-scene caption JSONs (auto-lookup target captions)")
    p.add_argument("--prompt", default=None,
                    help="Text prompt (overrides caption lookup if set)")
    # Shared
    p.add_argument("--H", type=int, default=256)
    p.add_argument("--W", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--cfg-text", type=float, default=3.0, help="Text CFG scale")
    p.add_argument("--seed", type=int, default=112)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    H, W = args.H, args.W
    lh, lw = H // 8, W // 8

    # --- Seed (same as eval_medium.sh) ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # --- Import arch-specific modules (same pattern as eval_medium.sh) ---
    if args.arch == "OLD":
        from css.old.pose_sd import PoseSD
        from css.old.scene_sampling import (
            build_comparison_grid, build_single_sample, load_scene_pools, to_uint8,
        )
    else:
        from css.models.pose_sd import PoseSD
        from css.inference.scene_sampling import (
            build_comparison_grid, build_single_sample, load_scene_pools, to_uint8,
        )

    print(f"Architecture: {args.arch}  Mode: {args.mode}")
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"Steps={args.num_steps}  CFG={args.cfg_scale}  Seed={args.seed}")

    # --- Load model (identical to eval_medium.sh) ---
    print("Loading model...")
    model = PoseSD()
    load_pose_sd_checkpoint(model, args.checkpoint, model.device)
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    # --- Resolve scene list ---
    if args.scene:
        scene_dirs = [Path(args.scene)]
    else:
        split_json = Path(args.split_dir) / "split_info.json"
        info = json.load(open(split_json))
        test_scenes = info["test_scenes"]
        selected = test_scenes[args.scene_idx : args.scene_idx + args.num_scenes]
        print(f"Test scenes available: {len(test_scenes)}, "
              f"using idx {args.scene_idx}..{args.scene_idx + len(selected) - 1} "
              f"({len(selected)} scenes)")
        data_root = Path(args.data_root)
        scene_dirs = [data_root / s for s in selected]

    out_dir = Path(args.output) if args.output else Path(f"outputs/{args.mode}_results2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Loop over scenes (same as eval_medium.sh) ---
    total = 0
    failed = 0

    for scene_dir in scene_dirs:
        if not scene_dir.exists():
            print(f"SKIP (not found): {scene_dir}")
            continue

        print(f"\n--- Scene: {scene_dir.name} ---")
        try:
            cameras, images_dir, target_images, reference_images = load_scene_pools(scene_dir)
        except Exception as e:
            print(f"  SKIP (load failed): {e}")
            continue

        # Load captions for this scene
        scene_captions = {}
        if args.caption_dir is not None:
            caption_file = Path(args.caption_dir) / f"{scene_dir.name}.json"
            if caption_file.exists():
                scene_captions = json.load(open(caption_file))

        if len(reference_images) < 2 or len(target_images) == 0:
            print(f"  SKIP (not enough images: {len(target_images)} targets, "
                  f"{len(reference_images)} refs)")
            continue

        triplets = find_medium_triplets(
            target_images, reference_images, max_triplets=args.targets_per_scene,
        )
        print(f"  Found {len(triplets)} MEDIUM-difficulty triplets")

        safe_scene = scene_dir.name[:60]
        scene_out = out_dir / safe_scene
        scene_out.mkdir(parents=True, exist_ok=True)

        for i, (ref1, ref2, tgt) in enumerate(triplets):
            # Resolve caption: --prompt overrides, otherwise look up from captions
            caption = args.prompt if args.prompt is not None else scene_captions.get(tgt.name, "")
            print(f"  [{i+1}/{len(triplets)}] {tgt.name}")
            try:
                sample = build_single_sample(cameras, images_dir, ref1, ref2, tgt, H, W)
                safe_tgt = Path(tgt.name).stem.replace("/", "_").replace(" ", "_")[:40]

                if args.mode == "eval":
                    results = generate_eval(
                        model, sample, args, to_uint8, build_comparison_grid,
                        caption=caption,
                    )
                else:  # sweep
                    results = generate_sweep(
                        model, sample, ref1, tgt, cameras, args,
                        to_uint8, build_cropped_scaled_intrinsics,
                        H, W, lh, lw, caption=caption,
                    )

                for label, grid in results:
                    out_path = scene_out / f"{safe_tgt}_{label}.png"
                    Image.fromarray(grid).save(out_path)

                total += 1
            except Exception as e:
                print(f"    FAILED: {e}")
                traceback.print_exc()
                failed += 1

    print(f"\n=== Done: {total} triplets, {failed} failed ===")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
