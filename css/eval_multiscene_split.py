"""Evaluate a multiscene split using test scenes."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from css.data.dataset import clean_scene_prompt_name, read_scene_prompt_name
from css.models.pose_conditioned_sd import PoseConditionedSD, load_pose_sd_checkpoint
from css.scene_sampling import (
    build_comparison_grid,
    build_single_sample,
    find_best_references,
    load_scene_pools,
)


@dataclass
class TargetRecord:
    target_idx: int
    scene_spec: str
    scene_path: Path
    scene_name: str
    image_id: int
    image_name: str
    prompt: str


def _read_lines(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _resolve_scene_path(scene_spec: str, scenes_root: Path | None) -> Path:
    path = Path(scene_spec)
    if path.is_absolute() or scenes_root is None:
        return path
    return scenes_root / path


def _scene_prompt(scene_dir: Path, prompt_template: str) -> str:
    scene_name = clean_scene_prompt_name(read_scene_prompt_name(scene_dir))
    if "{scene}" in prompt_template:
        return prompt_template.format(scene=scene_name)
    return prompt_template


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return slug[:80] if slug else "scene"


def _write_manifest(path: Path, records: list[TargetRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("target_idx\tscene_spec\tscene_name\timage_id\timage_name\tprompt\n")
        for r in records:
            f.write(
                f"{r.target_idx}\t{r.scene_spec}\t{r.scene_name}\t"
                f"{r.image_id}\t{r.image_name}\t{r.prompt}\n"
            )


def _write_skipped(path: Path, rows: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("scene_spec\treason\n")
        for scene_spec, reason in rows:
            f.write(f"{scene_spec}\t{reason}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split-dir", required=True, help="Directory containing test_scenes.txt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--scenes-root", default=None, help="Optional base path for relative scene entries")
    p.add_argument("--target-idx", type=int, default=None)
    p.add_argument("--all-targets", action="store_true")
    p.add_argument("--max-targets-per-scene", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-template", default="a photo of {scene}")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=7.5, help="CFG guidance scale")
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
    scenes_root = Path(args.scenes_root) if args.scenes_root is not None else None

    test_scenes_path = split_dir / "test_scenes.txt"
    if not test_scenes_path.exists():
        raise ValueError(f"Missing test scenes file: {test_scenes_path}")

    rng = np.random.default_rng(args.seed)
    scene_specs = _read_lines(test_scenes_path)
    records: list[TargetRecord] = []
    skipped: list[tuple[str, str]] = []

    for scene_spec in scene_specs:
        scene_dir = _resolve_scene_path(scene_spec, scenes_root)
        if not scene_dir.exists():
            skipped.append((scene_spec, "scene directory missing"))
            continue

        try:
            _, _, targets, refs = load_scene_pools(scene_dir)
        except Exception as e:
            skipped.append((scene_spec, f"load failed: {e}"))
            continue

        if len(targets) == 0:
            skipped.append((scene_spec, "no valid targets"))
            continue
        if len(refs) < 2:
            skipped.append((scene_spec, "fewer than 2 valid references"))
            continue

        keep = np.arange(len(targets))
        if args.max_targets_per_scene is not None and len(keep) > args.max_targets_per_scene:
            keep = np.sort(rng.choice(len(targets), size=args.max_targets_per_scene, replace=False))

        prompt = _scene_prompt(scene_dir, args.prompt_template)
        for local_idx in keep.tolist():
            target = targets[local_idx]
            records.append(
                TargetRecord(
                    target_idx=len(records),
                    scene_spec=scene_spec,
                    scene_path=scene_dir,
                    scene_name=scene_dir.name,
                    image_id=target.id,
                    image_name=target.name,
                    prompt=prompt,
                )
            )

    if len(records) == 0:
        skipped_path = output_dir / "skipped_scenes.tsv"
        _write_skipped(skipped_path, skipped)
        raise ValueError(f"No evaluable test targets found. See {skipped_path}")

    manifest_path = output_dir / "test_targets.tsv"
    _write_manifest(manifest_path, records)
    skipped_path = output_dir / "skipped_scenes.tsv"
    _write_skipped(skipped_path, skipped)
    print(f"Test scenes: {len(scene_specs)}")
    print(f"Evaluable targets: {len(records)}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote skipped scenes: {skipped_path}")

    if args.all_targets:
        target_indices = list(range(len(records)))
    else:
        if args.target_idx < 0 or args.target_idx >= len(records):
            raise ValueError(f"target_idx {args.target_idx} out of range [0, {len(records) - 1}]")
        target_indices = [args.target_idx]

    print("Loading model...")
    model = PoseConditionedSD()
    load_pose_sd_checkpoint(model, args.checkpoint, model.device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    cached_scene: Path | None = None
    cached_scene_data = None

    results_path = output_dir / "results.tsv"
    ok = 0
    failed = 0
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("target_idx\tscene_spec\timage_id\timage_name\tstatus\toutput\terror\n")
        for target_idx in target_indices:
            rec = records[target_idx]
            try:
                if cached_scene != rec.scene_path:
                    cameras, images_dir, targets, refs = load_scene_pools(rec.scene_path)
                    by_id = {img.id: img for img in targets}
                    by_name = {img.name: img for img in targets}
                    cached_scene = rec.scene_path
                    cached_scene_data = (cameras, images_dir, by_id, by_name, refs)

                cameras, images_dir, target_by_id, target_by_name, refs = cached_scene_data
                target_img = target_by_id.get(rec.image_id)
                if target_img is None:
                    target_img = target_by_name.get(rec.image_name)
                if target_img is None:
                    raise ValueError("Target disappeared after scene reload")

                ref1_img, ref2_img = find_best_references(
                    target_img,
                    refs,
                    max_dist=args.max_pair_dist,
                    min_dir_sim=args.min_dir_sim,
                    min_ref_spacing=args.min_ref_spacing,
                )
                sample = build_single_sample(
                    cameras,
                    images_dir,
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
                        sample["plucker_tgt"],
                        prompt=rec.prompt,
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
                    prompt=rec.prompt,
                )
                out_name = f"test_idx_{target_idx:06d}_{_safe_slug(rec.scene_name)}.png"
                output_path = output_dir / out_name
                Image.fromarray(grid).save(output_path)
                f.write(
                    f"{target_idx}\t{rec.scene_spec}\t{rec.image_id}\t{rec.image_name}\t"
                    f"ok\t{output_path}\t\n"
                )
                ok += 1
            except Exception as e:
                err = str(e).replace("\t", " ").replace("\n", " ")
                f.write(
                    f"{target_idx}\t{rec.scene_spec}\t{rec.image_id}\t{rec.image_name}\t"
                    f"fail\t\t{err}\n"
                )
                failed += 1
                print(f"Failed idx={target_idx}: {err}")

    print(f"Done. success={ok}, failed={failed}, total={len(target_indices)}")
    print(f"Wrote results: {results_path}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
