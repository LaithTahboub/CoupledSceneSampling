"""Evaluate split-held-out targets with train-only references."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from css.data.dataset import load_image_name_set
from css.models.pose_conditioned_sd import PoseConditionedSD, load_pose_sd_checkpoint
from css.scene_sampling import (
    build_comparison_grid,
    build_single_sample,
    find_best_references,
    load_scene_pools,
)


def _write_target_manifest(path: Path, targets) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("target_idx\timage_id\timage_name\n")
        for idx, img in enumerate(targets):
            f.write(f"{idx}\t{img.id}\t{img.name}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--split-dir", required=True, help="Directory containing train_images.txt and test_images.txt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-idx", type=int, default=None, help="Evaluate one test target by split index")
    p.add_argument("--all-targets", action="store_true", help="Evaluate all test targets")
    p.add_argument("--prompt", default="a photo of the Mysore palace")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--max-pair-dist", type=float, default=2.0)
    p.add_argument("--min-dir-sim", type=float, default=0.3)
    p.add_argument("--min-ref-spacing", type=float, default=0.3)
    p.add_argument("--noisy-target-start", action="store_true")
    p.add_argument("--start-t", type=int, default=500)
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)
    args = p.parse_args()

    if not args.all_targets and args.target_idx is None:
        args.target_idx = 0

    scene_dir = Path(args.scene)
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_list_path = split_dir / "train_images.txt"
    test_list_path = split_dir / "test_images.txt"
    if not train_list_path.exists() or not test_list_path.exists():
        raise ValueError(f"Missing split files under {split_dir}")

    train_names = load_image_name_set(str(train_list_path))
    test_names = load_image_name_set(str(test_list_path))

    cameras, images_dir, test_targets, train_refs = load_scene_pools(
        scene_dir,
        target_include_image_names=test_names,
        reference_include_image_names=train_names,
    )

    if len(test_targets) == 0:
        raise ValueError("No valid test targets after filtering")
    if len(train_refs) < 2:
        raise ValueError("Need at least 2 valid training references")

    manifest_path = output_dir / "test_targets.tsv"
    _write_target_manifest(manifest_path, test_targets)
    print(f"Test targets: {len(test_targets)} | Train refs: {len(train_refs)}")
    print(f"Wrote target index manifest: {manifest_path}")

    if args.all_targets:
        target_indices = list(range(len(test_targets)))
    else:
        if args.target_idx < 0 or args.target_idx >= len(test_targets):
            raise ValueError(f"target_idx {args.target_idx} out of range [0, {len(test_targets)-1}]")
        target_indices = [args.target_idx]

    print("Loading model...")
    model = PoseConditionedSD()
    load_pose_sd_checkpoint(model, args.checkpoint, model.device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    results_path = output_dir / "results.tsv"
    ok = 0
    failed = 0
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("target_idx\timage_id\timage_name\tstatus\toutput\terror\n")
        for idx in target_indices:
            target_img = test_targets[idx]
            output_path = output_dir / f"test_idx_{idx:05d}.png"
            print(f"[{idx}] {target_img.name}")

            try:
                ref1_img, ref2_img = find_best_references(
                    target_img,
                    train_refs,
                    max_dist=args.max_pair_dist,
                    min_dir_sim=args.min_dir_sim,
                    min_ref_spacing=args.min_ref_spacing,
                )
                sample = build_single_sample(cameras, images_dir, ref1_img, ref2_img, target_img, args.H, args.W)

                with torch.inference_mode():
                    generated = model.sample(
                        sample["ref1_img"],
                        sample["ref2_img"],
                        sample["plucker_ref1"],
                        sample["plucker_ref2"],
                        sample["plucker_target"],
                        prompt=args.prompt,
                        num_steps=args.num_steps,
                        cfg_scale=args.cfg_scale,
                        target=(sample["target_img"] if args.noisy_target_start else None),
                        start_t=args.start_t,
                    )

                grid = build_comparison_grid(
                    sample["ref1_img"][0],
                    sample["ref2_img"][0],
                    sample["target_img"][0],
                    generated[0],
                )
                Image.fromarray(grid).save(output_path)
                f.write(f"{idx}\t{target_img.id}\t{target_img.name}\tok\t{output_path}\t\n")
                ok += 1
            except Exception as e:
                err = str(e).replace("\t", " ").replace("\n", " ")
                f.write(f"{idx}\t{target_img.id}\t{target_img.name}\tfail\t\t{err}\n")
                failed += 1
                print(f"Failed idx={idx}: {err}")

    print(f"Done. success={ok}, failed={failed}, total={len(target_indices)}")
    print(f"Wrote results: {results_path}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
