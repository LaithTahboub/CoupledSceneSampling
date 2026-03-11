"""Sample from a PoseSD checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from css.data.dataset import (
    clean_scene_prompt_name,
    read_scene_prompt_name,
)
from css.models.EMA import load_pose_sd_checkpoint
from css.models.pose_sd import PoseSD
from css.scene_sampling import (
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
    parser.add_argument("--prompt", default=None, help="Text prompt (auto-generated from scene name if not set)")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
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
    if args.prompt is None:
        scene_text = clean_scene_prompt_name(read_scene_prompt_name(scene_dir))
        prompt = f"a photo of {scene_text}"
    else:
        prompt = args.prompt

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

    # Select target
    if args.target is not None:
        matches = [img for img in target_images if img.name == args.target or Path(img.name).name == args.target]
        if not matches:
            raise ValueError(f"Target '{args.target}' not found in scene")
        target_img = matches[0]
    else:
        target_img = target_images[np.random.randint(0, len(target_images))]

    print(f"Target: {target_img.name}")

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
