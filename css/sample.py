"""Sample from a pose-conditioned SD checkpoint."""

import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from css.data.colmap_reader import read_scene
from css.models.pose_conditioned_sd import PoseConditionedSD, load_pose_sd_checkpoint
from css.data.dataset import MegaScenesDataset, load_image_name_set


def compute_viewing_direction(c2w: np.ndarray) -> np.ndarray:
    return -c2w[:3, 2]


def find_best_references(target_img, all_images, max_dist=2.0, min_dir_sim=0.3, min_ref_spacing=0.3):
    target_pos = target_img.c2w[:3, 3]
    target_dir = compute_viewing_direction(target_img.c2w)

    candidates = []
    for img in all_images:
        if img.id == target_img.id:
            continue

        pos = img.c2w[:3, 3]
        dir = compute_viewing_direction(img.c2w)

        distance = np.linalg.norm(pos - target_pos)
        if distance > max_dist:
            continue

        dir_sim = np.dot(target_dir, dir)
        if dir_sim < min_dir_sim:
            continue

        score = distance * (2.0 - dir_sim)
        candidates.append((score, img, distance, dir_sim))

    if len(candidates) < 2:
        raise ValueError(f"Not enough valid references found for target {target_img.name}")

    candidates.sort(key=lambda x: x[0])

    ref1_score, ref1, ref1_dist, ref1_sim = candidates[0]

    ref2 = None
    ref2_score, ref2_dist, ref2_sim = None, None, None
    for score, img, dist, sim in candidates[1:]:
        ref1_to_ref2 = np.linalg.norm(img.c2w[:3, 3] - ref1.c2w[:3, 3])
        if ref1_to_ref2 >= min_ref_spacing:
            ref2 = img
            ref2_score, ref2_dist, ref2_sim = score, dist, sim
            break

    if ref2 is None:
        ref2_score, ref2, ref2_dist, ref2_sim = candidates[1]

    print(f"Selected references:")
    print(f"  Ref1: {ref1.name} (dist={ref1_dist:.2f}, dir_sim={ref1_sim:.2f}, score={ref1_score:.2f})")
    print(f"  Ref2: {ref2.name} (dist={ref2_dist:.2f}, dir_sim={ref2_sim:.2f}, score={ref2_score:.2f})")

    return ref1, ref2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to UNet checkpoint (.pt file)")
    parser.add_argument("--scene", default="MegaScenes/Mysore_Palace", help="Scene directory")
    parser.add_argument("--target-idx", type=int, default=None, help="Target image index (or random if not set)")
    parser.add_argument("--prompt", default="a photo of the Mysore palace", help="Text prompt")
    parser.add_argument("--num-steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="CFG scale")
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
    args = parser.parse_args()

    scene_dir = Path(args.scene)

    print("Loading model...")
    model = PoseConditionedSD()
    load_pose_sd_checkpoint(model, args.checkpoint, model.device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    print(f"Loading scene: {scene_dir}")
    cameras, images = read_scene(scene_dir)
    images_dir = scene_dir / "images"

    exclude_image_names = load_image_name_set(args.exclude_image_list)
    target_include_image_names = load_image_name_set(args.target_include_image_list)
    reference_include_image_names = load_image_name_set(args.reference_include_image_list)

    valid_images = [img for img in images.values() if (images_dir / img.name).exists()]
    valid_images.sort(key=lambda x: x.id)
    if exclude_image_names is not None:
        valid_images = [img for img in valid_images if img.name not in exclude_image_names]
    print(f"Found {len(valid_images)} valid images")

    target_images = valid_images
    if target_include_image_names is not None:
        target_images = [img for img in target_images if img.name in target_include_image_names]
        print(f"Target pool after filter: {len(target_images)}")

    reference_images = valid_images
    if reference_include_image_names is not None:
        reference_images = [img for img in reference_images if img.name in reference_include_image_names]
        print(f"Reference pool after filter: {len(reference_images)}")

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

    dataset = MegaScenesDataset([str(scene_dir)], H=512, W=512, max_triplets_per_scene=1)

    ref1_tensor = dataset._load_image(images_dir, ref1_img).unsqueeze(0)
    ref2_tensor = dataset._load_image(images_dir, ref2_img).unsqueeze(0)
    target_tensor = dataset._load_image(images_dir, target_img).unsqueeze(0)

    cam_ref1 = cameras[ref1_img.camera_id]
    cam_ref2 = cameras[ref2_img.camera_id]
    cam_tgt = cameras[target_img.camera_id]

    K_ref1 = dataset._build_K(cam_ref1)
    K_ref2 = dataset._build_K(cam_ref2)
    K_tgt = dataset._build_K(cam_tgt)

    plucker_ref1 = dataset._compute_plucker(ref1_img.c2w, ref1_img.c2w, K_ref1).unsqueeze(0)
    plucker_ref2 = dataset._compute_plucker(ref1_img.c2w, ref2_img.c2w, K_ref2).unsqueeze(0)
    plucker_target = dataset._compute_plucker(ref1_img.c2w, target_img.c2w, K_tgt).unsqueeze(0)

    print(f"\nGenerating with {args.num_steps} steps, CFG={args.cfg_scale}...")
    with torch.inference_mode():
        
        generated = model.sample(
            ref1_tensor, ref2_tensor,
            plucker_ref1, plucker_ref2, plucker_target,
            prompt=args.prompt, num_steps=args.num_steps, cfg_scale=args.cfg_scale, 
            target=(target_tensor if args.noisy_target_start else None), start_t=args.start_t
        )

    def to_pil(t):
        arr = ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)

    ref1_pil = to_pil(ref1_tensor[0])
    ref2_pil = to_pil(ref2_tensor[0])
    target_pil = to_pil(target_tensor[0])
    generated_pil = to_pil(generated[0])

    grid = np.concatenate([
        np.array(ref1_pil),
        np.array(ref2_pil),
        np.array(target_pil),
        np.array(generated_pil),
    ], axis=1)

    output_path = Path(args.output)
    Image.fromarray(grid).save(output_path)
    print(f"Saved comparison: {output_path}")
    print("  [Ref1 | Ref2 | Ground Truth | Generated]")

    if args.show_refs:
        ref1_pil.save(output_path.with_stem(output_path.stem + "_ref1"))
        ref2_pil.save(output_path.with_stem(output_path.stem + "_ref2"))
        target_pil.save(output_path.with_stem(output_path.stem + "_gt"))
        generated_pil.save(output_path.with_stem(output_path.stem + "_gen"))
        print(f"Saved individual images")


if __name__ == "__main__":
    main()
