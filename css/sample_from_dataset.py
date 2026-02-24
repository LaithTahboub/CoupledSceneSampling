"""Sample with dataset triplets."""

import argparse
from pathlib import Path
from collections import OrderedDict

import torch
from PIL import Image

from css.data.dataset import MegaScenesDataset, load_image_name_set
from css.models.pose_conditioned_sd import PoseConditionedSD, load_pose_sd_checkpoint
from css.scene_sampling import build_comparison_grid


def _read_lines(path: str | None) -> list[str]:
    if path is None:
        return []
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to UNet checkpoint (.pt file)")
    parser.add_argument("--scene", default=None, help="One scene directory")
    parser.add_argument("--scenes-file", type=str, default=None, help="Optional list of scenes (one per line)")
    parser.add_argument("--triplet-idx", type=int, default=0, help="Which dataset triplet to use")
    parser.add_argument("--prompt", default="", help="Text prompt")
    parser.add_argument("--prompt-template", type=str, default=None, help='Optional template, e.g. "a photo of {scene}"')
    parser.add_argument("--num-steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--max-pair-dist", type=float, default=2.0)
    parser.add_argument("--max-triplets", type=int, default=8)
    parser.add_argument("--triplets-manifest", type=str, default=None, help="Optional jsonl manifest of packed triplets")
    parser.add_argument("--exclude-image-list", type=str, default=None)
    parser.add_argument("--target-include-image-list", type=str, default=None)
    parser.add_argument("--reference-include-image-list", type=str, default=None)
    parser.add_argument("--output", default="sample_ds.png", help="Output path")
    parser.add_argument("--start-t", type=int, default=500, help="t value for noisy-target start")
    parser.add_argument("--noisy-target-start", action="store_true")
    args = parser.parse_args()

    print("Loading model...")
    model = PoseConditionedSD()
    load_pose_sd_checkpoint(model, args.checkpoint, model.device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    scenes = _read_lines(args.scenes_file)
    if args.scene is not None:
        scenes = [args.scene] + scenes
    if not scenes and args.triplets_manifest is None:
        scenes = ["MegaScenes/Mysore_Palace"]
    scenes = list(OrderedDict.fromkeys(scenes))

    if args.triplets_manifest is not None:
        print(f"Loading dataset from manifest: {args.triplets_manifest}")
    else:
        print(f"Loading dataset from {len(scenes)} scene(s)...")
    exclude_image_names = load_image_name_set(args.exclude_image_list)
    target_include_image_names = load_image_name_set(args.target_include_image_list)
    reference_include_image_names = load_image_name_set(args.reference_include_image_list)

    dataset = MegaScenesDataset(
        scenes,
        H=512, W=512,
        max_pair_distance=args.max_pair_dist,
        max_triplets_per_scene=args.max_triplets,
        exclude_image_names=exclude_image_names,
        target_include_image_names=target_include_image_names,
        reference_include_image_names=reference_include_image_names,
        prompt_template=args.prompt_template,
        triplets_manifest=args.triplets_manifest,
    )
    print(f"Dataset has {len(dataset)} triplets")
    if len(dataset) == 0:
        raise ValueError("Dataset has 0 triplets. Check split files and triplet constraints.")

    if args.triplet_idx >= len(dataset):
        print(f"Error: triplet_idx {args.triplet_idx} out of range (max: {len(dataset)-1})")
        return

    sample = dataset[args.triplet_idx]
    print(f"\nUsing triplet {args.triplet_idx}")
    sample_prompt = args.prompt if args.prompt else sample.get("prompt", "")
    print(f'Using prompt: "{sample_prompt}"')

    print(f"Generating with {args.num_steps} steps, cfg_scale={args.cfg_scale}...")
    with torch.inference_mode():
        generated = model.sample(
            sample["ref1_img"].unsqueeze(0),
            sample["ref2_img"].unsqueeze(0),
            sample["plucker_ref1"].unsqueeze(0),
            sample["plucker_ref2"].unsqueeze(0),
            sample["plucker_tgt"].unsqueeze(0),
            prompt=sample_prompt,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            target=(sample["target_img"].unsqueeze(0) if args.noisy_target_start else None),
            start_t=args.start_t,
        )

    grid = build_comparison_grid(
        sample["ref1_img"],
        sample["ref2_img"],
        sample["target_img"],
        generated[0],
        prompt=sample_prompt,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)
    print(f"\nSaved: {output_path}")
    print("  [Ref1 | Ref2 | Ground Truth | Generated]")


if __name__ == "__main__":
    main()
