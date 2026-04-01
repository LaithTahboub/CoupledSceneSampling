"""Sample from a PoseSD checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from css.models.EMA import load_pose_sd_checkpoint
from css.models.pose_sd import PoseSD
from css.inference.scene_sampling import (
    build_comparison_grid,
    build_single_sample,
    find_best_references,
    load_scene_pools,
    to_uint8,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to UNet checkpoint (.pt file)")
    parser.add_argument("--scene", required=True, help="Scene directory")
    parser.add_argument("--target", type=str, default=None, help="Target image name (or random if not set)")
    parser.add_argument("--ref1", type=str, default=None, help="Ref1 image name (skip auto-selection)")
    parser.add_argument("--ref2", type=str, default=None, help="Ref2 image name (skip auto-selection)")
    parser.add_argument("--prompt", default=None, help="Text prompt (overrides caption lookup if set)")
    parser.add_argument("--caption-dir", default=None, help="Directory with per-scene caption JSONs")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--cfg-text", type=float, default=3.0, help="Text CFG scale")
    parser.add_argument("--min-covisibility", type=float, default=0.15)
    parser.add_argument("--max-covisibility", type=float, default=0.80)
    parser.add_argument("--min-distance", type=float, default=0.20)
    parser.add_argument("--exclude-images", nargs="*", default=None, help="Image names to exclude from ref pool")
    parser.add_argument("--output", default="sample.png")
    parser.add_argument("--show-pluckers", action="store_true")
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scene_dir = Path(args.scene)

    # Prompt resolution: --prompt overrides everything, otherwise look up
    # the target's caption from the caption dir, falling back to "".
    prompt = args.prompt  # None means "auto from captions"
    _scene_captions = {}
    if prompt is None and args.caption_dir is not None:
        caption_file = Path(args.caption_dir) / f"{scene_dir.name}.json"
        if caption_file.exists():
            _scene_captions = json.load(open(caption_file))
    if prompt is None:
        prompt = ""  # resolved after target is selected, below

    print("Loading model...")
    model = PoseSD()
    load_pose_sd_checkpoint(model, args.checkpoint, model.device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    exclude_set = set(args.exclude_images) if args.exclude_images else None

    print(f"Loading scene: {scene_dir}")
    cameras, images_dir, target_images, reference_images = load_scene_pools(
        scene_dir, exclude_image_names=exclude_set,
    )
    print(f"Target pool: {len(target_images)} | Reference pool: {len(reference_images)}")

    if len(target_images) == 0:
        raise ValueError("No target images available")
    if len(reference_images) < 2:
        raise ValueError("Need at least 2 reference images")

    seen = set()
    all_images = []
    for img in target_images + reference_images:
        if img.name not in seen:
            seen.add(img.name)
            all_images.append(img)

    def _find_image(name: str, pool: list) -> "ImageData":
        matches = [img for img in pool if img.name == name or Path(img.name).name == name]
        if not matches:
            raise ValueError(f"Image '{name}' not found in scene")
        return matches[0]

    # Select target
    if args.target is not None:
        target_img = _find_image(args.target, target_images)
    else:
        target_img = target_images[np.random.randint(0, len(target_images))]

    print(f"Target: {target_img.name}")

    # Resolve caption if not explicitly set via --prompt
    if args.prompt is None and _scene_captions:
        prompt = _scene_captions.get(target_img.name, "")

    # Select references
    if args.ref1 is not None and args.ref2 is not None:
        ref1_img = _find_image(args.ref1, all_images)
        ref2_img = _find_image(args.ref2, all_images)
    elif args.ref1 is not None or args.ref2 is not None:
        raise ValueError("Provide both --ref1 and --ref2, or neither")
    else:
        ref1_img, ref2_img = find_best_references(
            target_img, reference_images,
            min_covisibility=args.min_covisibility,
            max_covisibility=args.max_covisibility,
            min_distance=args.min_distance,
        )
    print(f"Refs: {ref1_img.name}, {ref2_img.name}")

    sample = build_single_sample(cameras, images_dir, ref1_img, ref2_img, target_img, args.H, args.W)

    print(f'Generating ({args.num_steps} steps, cfg={args.cfg_scale}, prompt="{prompt}")')
    with torch.inference_mode():
        generated = model.sample(
            ref1_img=sample["ref1_img"],
            ref2_img=sample["ref2_img"],
            pl_ref1=sample["plucker_ref1"],
            pl_ref2=sample["plucker_ref2"],
            pl_tgt=sample["plucker_tgt"],
            prompt=prompt,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            cfg_text=args.cfg_text,
            seed=args.seed,
        )

    pluckers = None
    if args.show_pluckers:
        pluckers = (sample["plucker_ref1"][0], sample["plucker_ref2"][0], sample["plucker_tgt"][0])

    grid = build_comparison_grid(
        sample["ref1_img"][0], sample["ref2_img"][0],
        sample["target_img"][0], generated[0],
        prompt=prompt, pluckers=pluckers,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)
    print(f"Saved: {output_path}  [Ref1 | Ref2 | GT | Generated]")


if __name__ == "__main__":
    main()
