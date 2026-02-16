"""Evaluate held-out test targets across multiple scenes using train-only references."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from css.data.dataset import load_image_name_set
from css.data.scene_names import derive_scene_key, scene_prompt_name
from css.models.pose_conditioned_sd import PoseConditionedSD, load_pose_sd_checkpoint
from css.scene_sampling import (
    build_comparison_grid,
    build_single_sample,
    find_best_references,
    load_scene_pools,
)


def _read_keys(path: Path) -> list[str]:
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _split_scene_key(scene_key: str) -> tuple[str, str]:
    if "\t" in scene_key:
        scene_name, image_name = scene_key.split("\t", 1)
        return scene_name, image_name
    if "/" in scene_key:
        scene_name, image_name = scene_key.split("/", 1)
        return scene_name, image_name
    raise ValueError(f"Expected scene-qualified key, got: {scene_key}")


def _render_prompt(template: str, scene_name: str) -> str:
    scene_text = scene_prompt_name(scene_name)
    if "{scene}" not in template:
        return template
    return template.format(scene=scene_text)


def _write_target_manifest(path: Path, scene_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("target_idx\tscene_name\timage_name\tscene_key\n")
        for idx, key in enumerate(scene_keys):
            scene_name, image_name = _split_scene_key(key)
            f.write(f"{idx}\t{scene_name}\t{image_name}\t{key}\n")


def _load_scene_map(scenes_file: Path) -> dict[str, Path]:
    scene_map: dict[str, Path] = {}
    for line in _read_keys(scenes_file):
        scene_dir = Path(line)
        scene_name = derive_scene_key(scene_dir)
        if scene_name in scene_map and scene_map[scene_name] != scene_dir:
            raise ValueError(f"Duplicate scene key '{scene_name}' in {scenes_file}")
        scene_map[scene_name] = scene_dir
    if not scene_map:
        raise ValueError(f"No scenes found in {scenes_file}")
    return scene_map


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split-dir", required=True, help="Directory with scenes.txt/train_images.txt/test_images.txt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-idx", type=int, default=None, help="Evaluate one held-out target by global split index")
    p.add_argument("--all-targets", action="store_true", help="Evaluate all held-out targets")
    p.add_argument("--prompt-template", default="a photo of {scene}")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=1.0, help="Guidance scale (APG)")
    p.add_argument("--apg-eta", type=float, default=0.0)
    p.add_argument("--apg-momentum", type=float, default=-0.5)
    p.add_argument("--apg-norm-threshold", type=float, default=0.0)
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

    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes_file = split_dir / "scenes.txt"
    train_list_path = split_dir / "train_images.txt"
    test_list_path = split_dir / "test_images.txt"
    for pth in (scenes_file, train_list_path, test_list_path):
        if not pth.exists():
            raise ValueError(f"Missing required split file: {pth}")

    train_names = load_image_name_set(str(train_list_path))
    test_names = load_image_name_set(str(test_list_path))
    test_keys = _read_keys(test_list_path)
    if not test_keys:
        raise ValueError("No test targets found in split")

    scene_map = _load_scene_map(scenes_file)
    _write_target_manifest(output_dir / "test_targets.tsv", test_keys)

    if args.all_targets:
        target_indices = list(range(len(test_keys)))
    else:
        if args.target_idx < 0 or args.target_idx >= len(test_keys):
            raise ValueError(f"target_idx {args.target_idx} out of range [0, {len(test_keys)-1}]")
        target_indices = [args.target_idx]

    print("Loading model...")
    model = PoseConditionedSD()
    load_pose_sd_checkpoint(model, args.checkpoint, model.device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    scene_cache: dict[str, dict] = {}

    def get_scene_data(scene_name: str) -> dict:
        if scene_name in scene_cache:
            return scene_cache[scene_name]
        scene_dir = scene_map.get(scene_name)
        if scene_dir is None:
            raise ValueError(f"Scene '{scene_name}' not found in {scenes_file}")
        cameras, images_dir, test_targets, train_refs = load_scene_pools(
            scene_dir,
            target_include_image_names=test_names,
            reference_include_image_names=train_names,
        )
        target_by_name = {img.name: img for img in test_targets}
        if len(train_refs) < 2:
            raise ValueError(f"{scene_name}: need at least 2 train references")
        out = {
            "cameras": cameras,
            "images_dir": images_dir,
            "target_by_name": target_by_name,
            "train_refs": train_refs,
        }
        scene_cache[scene_name] = out
        return out

    results_path = output_dir / "results.tsv"
    ok = 0
    failed = 0
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("target_idx\tscene_name\timage_name\tstatus\toutput\terror\n")
        for idx in target_indices:
            scene_name, image_name = _split_scene_key(test_keys[idx])
            print(f"[{idx}] {scene_name}/{image_name}")
            scene_out_dir = output_dir / scene_name.replace("/", "__")
            scene_out_dir.mkdir(parents=True, exist_ok=True)
            output_path = scene_out_dir / f"test_idx_{idx:06d}.png"

            try:
                scene_data = get_scene_data(scene_name)
                target_img = scene_data["target_by_name"].get(image_name)
                if target_img is None:
                    raise ValueError(f"Target not found in scene pools: {scene_name}/{image_name}")

                ref1_img, ref2_img = find_best_references(
                    target_img,
                    scene_data["train_refs"],
                    max_dist=args.max_pair_dist,
                    min_dir_sim=args.min_dir_sim,
                    min_ref_spacing=args.min_ref_spacing,
                )
                sample = build_single_sample(
                    scene_data["cameras"],
                    scene_data["images_dir"],
                    ref1_img,
                    ref2_img,
                    target_img,
                    args.H,
                    args.W,
                )

                with torch.inference_mode():
                    generated = model.sample(
                        sample["ref1_img"],
                        sample["ref2_img"],
                        sample["plucker_ref1"],
                        sample["plucker_ref2"],
                        sample["plucker_target"],
                        prompt=_render_prompt(args.prompt_template, scene_name),
                        num_steps=args.num_steps,
                        cfg_scale=args.cfg_scale,
                        apg_eta=args.apg_eta,
                        apg_momentum=args.apg_momentum,
                        apg_norm_threshold=args.apg_norm_threshold,
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
                f.write(f"{idx}\t{scene_name}\t{image_name}\tok\t{output_path}\t\n")
                ok += 1
            except Exception as e:
                err = str(e).replace("\t", " ").replace("\n", " ")
                f.write(f"{idx}\t{scene_name}\t{image_name}\tfail\t\t{err}\n")
                failed += 1
                print(f"Failed idx={idx}: {err}")

    print(f"Done. success={ok}, failed={failed}, total={len(target_indices)}")
    print(f"Wrote results: {results_path}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
