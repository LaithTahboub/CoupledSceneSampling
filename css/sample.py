"""Sample from a pose-conditioned SD checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from css.data.dataset import (
    clean_scene_prompt_name,
    load_image_name_set,
    read_scene_prompt_name,
)
from css.models.pose_conditioned_sd import PoseConditionedSD, load_pose_sd_checkpoint
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
    parser.add_argument("--scene", default="MegaScenes/Mysore_Palace", help="Scene directory")
    parser.add_argument("--target-idx", type=int, default=None, help="Target image index (or random if not set)")
    parser.add_argument("--prompt", default="a photo of the Mysore palace", help="Text prompt")
    parser.add_argument("--prompt-template", type=str, default=None, help='Optional template, e.g. "a photo of {scene}"')
    parser.add_argument("--num-steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--max-pair-dist", type=float, default=2.0, help="Max ref-target camera distance")
    parser.add_argument("--min-dir-sim", type=float, default=0.3, help="Min view direction similarity")
    parser.add_argument("--min-ref-spacing", type=float, default=0.3, help="Min distance between refs")
    parser.add_argument("--exclude-image-list", type=str, default=None)
    parser.add_argument("--target-include-image-list", type=str, default=None)
    parser.add_argument("--reference-include-image-list", type=str, default=None)
    parser.add_argument("--output", default="sample.png", help="Output path")
    parser.add_argument("--noisy-target-start", action="store_true")
    parser.add_argument("--show-refs", action="store_true", help="Also save reference images")
    parser.add_argument("--start-t", type=int, default=500, help="t value for noisy-target start")
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    args = parser.parse_args()

    scene_dir = Path(args.scene)
    prompt = args.prompt
    if args.prompt_template is not None and args.prompt == parser.get_default("prompt"):
        scene_text = clean_scene_prompt_name(read_scene_prompt_name(scene_dir))
        prompt = args.prompt_template.format(scene=scene_text) if "{scene}" in args.prompt_template else args.prompt_template

    print("Loading model...")
    model = PoseConditionedSD()
    load_pose_sd_checkpoint(model, args.checkpoint, model.device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    exclude_image_names = load_image_name_set(args.exclude_image_list)
    target_include_image_names = load_image_name_set(args.target_include_image_list)
    reference_include_image_names = load_image_name_set(args.reference_include_image_list)

    print(f"Loading scene: {scene_dir}")
    cameras, images_dir, target_images, reference_images = load_scene_pools(
        scene_dir,
        exclude_image_names=exclude_image_names,
        target_include_image_names=target_include_image_names,
        reference_include_image_names=reference_include_image_names,
    )
    print(f"Target pool: {len(target_images)} | Reference pool: {len(reference_images)}")

    if len(target_images) == 0:
        raise ValueError("No target images available after filtering")
    if len(reference_images) < 2:
        raise ValueError("Need at least 2 reference images after filtering")

    if args.target_idx is not None:
        if args.target_idx < 0 or args.target_idx >= len(target_images):
            raise ValueError(f"target_idx {args.target_idx} out of range [0, {len(target_images)-1}]")
        target_img = target_images[args.target_idx]
    else:
        idx = np.random.randint(0, len(target_images))
        target_img = target_images[idx]

    print(f"\nTarget: {target_img.name} (image {target_img.id})")

    ref1_img, ref2_img = find_best_references(
        target_img,
        reference_images,
        max_dist=args.max_pair_dist,
        min_dir_sim=args.min_dir_sim,
        min_ref_spacing=args.min_ref_spacing,
    )
    print(f"Selected refs: {ref1_img.name}, {ref2_img.name}")

    sample = build_single_sample(cameras, images_dir, ref1_img, ref2_img, target_img, args.H, args.W)

    print(f'\nGenerating with {args.num_steps} steps, cfg_scale={args.cfg_scale}, prompt="{prompt}"')
    with torch.inference_mode():
        generated = model.sample(
            sample["ref1_img"], sample["ref2_img"],
            sample["plucker_ref1"], sample["plucker_ref2"],
            prompt=prompt,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            target=(sample["target_img"] if args.noisy_target_start else None),
            start_t=args.start_t,
        )

    grid = build_comparison_grid(sample["ref1_img"][0], sample["ref2_img"][0], sample["target_img"][0], generated[0])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)
    print(f"Saved comparison: {output_path}")
    print("  [Ref1 | Ref2 | Ground Truth | Generated]")

    if args.show_refs:
        Image.fromarray(to_uint8(sample["ref1_img"][0])).save(output_path.with_stem(output_path.stem + "_ref1"))
        Image.fromarray(to_uint8(sample["ref2_img"][0])).save(output_path.with_stem(output_path.stem + "_ref2"))
        Image.fromarray(to_uint8(sample["target_img"][0])).save(output_path.with_stem(output_path.stem + "_gt"))
        Image.fromarray(to_uint8(generated[0])).save(output_path.with_stem(output_path.stem + "_gen"))
        print("Saved individual images")


if __name__ == "__main__":
    main()
