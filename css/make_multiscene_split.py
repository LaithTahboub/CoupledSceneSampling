"""Create train/test splits across multiple scenes with scene-qualified image keys."""

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from css.data.colmap_reader import ImageData, read_scene
from css.data.scene_names import derive_scene_key, scene_pair_key


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


def _resolve_scenes(args) -> list[str]:
    merged = list(args.scenes or []) + _read_lines(args.scenes_file)
    if not merged:
        raise ValueError("Provide at least one scene via --scenes or --scenes-file")
    return list(OrderedDict.fromkeys(merged))


def _valid_image_names(scene_dir: Path) -> list[str]:
    _, images = read_scene(scene_dir)
    images_dir = scene_dir / "images"
    names = [img.name for img in images.values() if (images_dir / img.name).exists()]
    return sorted(set(names))


def _view_dir(c2w: np.ndarray) -> np.ndarray:
    return -c2w[:3, 2]


def _pose_is_valid(c2w: np.ndarray, max_rot_orth_err: float, max_det_err: float) -> bool:
    if not np.isfinite(c2w).all():
        return False
    R = c2w[:3, :3]
    det_err = abs(float(np.linalg.det(R)) - 1.0)
    orth_err = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
    return det_err <= max_det_err and orth_err <= max_rot_orth_err


def _has_valid_ref_pair(
    target: ImageData,
    pool: list[ImageData],
    *,
    max_pair_dist: float,
    min_dir_sim: float,
    min_ref_spacing: float,
) -> bool:
    target_pos = target.c2w[:3, 3]
    target_dir = _view_dir(target.c2w)

    candidates: list[tuple[float, ImageData]] = []
    for ref in pool:
        if ref.id == target.id:
            continue
        dist = float(np.linalg.norm(ref.c2w[:3, 3] - target_pos))
        if dist > max_pair_dist:
            continue
        sim = float(np.dot(target_dir, _view_dir(ref.c2w)))
        if sim < min_dir_sim:
            continue
        candidates.append((dist * (2.0 - sim), ref))

    if len(candidates) < 2:
        return False

    candidates.sort(key=lambda x: x[0])
    top = [x[1] for x in candidates[: min(20, len(candidates))]]
    for i in range(len(top) - 1):
        for j in range(i + 1, len(top)):
            spacing = float(np.linalg.norm(top[i].c2w[:3, 3] - top[j].c2w[:3, 3]))
            if spacing >= min_ref_spacing:
                return True
    return False


def _pose_quality(scene_dir: Path, args) -> tuple[bool, dict]:
    _, images = read_scene(scene_dir)
    images_dir = scene_dir / "images"
    all_images = [img for img in images.values() if (images_dir / img.name).exists()]
    if len(all_images) < 3:
        return False, {
            "reason": f"too_few_images:{len(all_images)}",
            "usable_images": len(all_images),
            "pose_valid_images": 0,
            "coverage": 0.0,
            "covered_targets": 0,
        }

    pose_valid = [
        img
        for img in all_images
        if _pose_is_valid(img.c2w, args.quality_max_rot_orth_err, args.quality_max_det_err)
    ]
    if len(pose_valid) < 3:
        return False, {
            "reason": f"too_few_pose_valid:{len(pose_valid)}",
            "usable_images": len(all_images),
            "pose_valid_images": len(pose_valid),
            "coverage": 0.0,
            "covered_targets": 0,
        }

    pose_valid_ratio = len(pose_valid) / max(1, len(all_images))
    if pose_valid_ratio < args.quality_min_pose_valid_ratio:
        return False, {
            "reason": f"low_pose_valid_ratio:{pose_valid_ratio:.3f}",
            "usable_images": len(all_images),
            "pose_valid_images": len(pose_valid),
            "coverage": 0.0,
            "covered_targets": 0,
        }

    covered_targets = 0
    for target in pose_valid:
        if _has_valid_ref_pair(
            target,
            pose_valid,
            max_pair_dist=args.quality_max_pair_dist,
            min_dir_sim=args.quality_min_dir_sim,
            min_ref_spacing=args.quality_min_ref_spacing,
        ):
            covered_targets += 1

    coverage = covered_targets / max(1, len(pose_valid))
    if covered_targets < args.quality_min_covered_targets:
        return False, {
            "reason": f"covered_targets<{args.quality_min_covered_targets}",
            "usable_images": len(all_images),
            "pose_valid_images": len(pose_valid),
            "coverage": coverage,
            "covered_targets": covered_targets,
        }
    if coverage < args.quality_min_coverage:
        return False, {
            "reason": f"coverage<{args.quality_min_coverage}",
            "usable_images": len(all_images),
            "pose_valid_images": len(pose_valid),
            "coverage": coverage,
            "covered_targets": covered_targets,
        }

    return True, {
        "reason": "ok",
        "usable_images": len(all_images),
        "pose_valid_images": len(pose_valid),
        "coverage": coverage,
        "covered_targets": covered_targets,
    }


def _stable_seed(scene_name: str, base_seed: int) -> int:
    digest = hashlib.blake2b(scene_name.encode("utf-8"), digest_size=8).digest()
    scene_seed = int.from_bytes(digest, byteorder="little", signed=False)
    return (base_seed ^ scene_seed) % (2**63 - 1)


def _split_scene_names(
    names: list[str],
    *,
    test_ratio: float,
    train_ratio: float,
    seed: int,
    min_train_images: int,
) -> tuple[list[str], list[str]]:
    n = len(names)
    if n < min_train_images + 1:
        raise ValueError(f"Need at least {min_train_images + 1} images, found {n}")

    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("--test-ratio must be in [0, 1)")
    if not (0.0 < train_ratio <= 1.0):
        raise ValueError("--train-ratio must be in (0, 1]")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    n_test = int(round(n * test_ratio))
    if test_ratio > 0 and n_test == 0:
        n_test = 1
    n_test = min(n_test, n - min_train_images)

    test_idx = set(perm[:n_test].tolist())
    train_names_all = [name for i, name in enumerate(names) if i not in test_idx]
    test_names = [name for i, name in enumerate(names) if i in test_idx]

    n_train_keep = int(round(len(train_names_all) * train_ratio))
    n_train_keep = max(min_train_images, min(len(train_names_all), n_train_keep))
    train_perm = rng.permutation(len(train_names_all))
    keep_idx = set(train_perm[:n_train_keep].tolist())
    train_names = [name for i, name in enumerate(train_names_all) if i in keep_idx]

    train_names.sort()
    test_names.sort()
    overlap = set(train_names).intersection(test_names)
    if overlap:
        raise RuntimeError(f"Split bug: {len(overlap)} images overlap between train/test")
    return train_names, test_names


def _write_names(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if names:
        path.write_text("\n".join(names) + "\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--scenes-file", type=str, default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--train-ratio", type=float, default=1.0, help="Fraction of non-test images kept for training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-train-images", type=int, default=3)
    p.add_argument("--min-scenes", type=int, default=1)
    p.add_argument("--pose-quality-filter", action="store_true")
    p.add_argument("--quality-max-pair-dist", type=float, default=2.5)
    p.add_argument("--quality-min-dir-sim", type=float, default=0.2)
    p.add_argument("--quality-min-ref-spacing", type=float, default=0.25)
    p.add_argument("--quality-max-rot-orth-err", type=float, default=0.05)
    p.add_argument("--quality-max-det-err", type=float, default=0.10)
    p.add_argument("--quality-min-pose-valid-ratio", type=float, default=0.95)
    p.add_argument("--quality-min-covered-targets", type=int, default=20)
    p.add_argument("--quality-min-coverage", type=float, default=0.35)
    p.add_argument("--strict", action="store_true", help="Fail if any scene is invalid")
    args = p.parse_args()

    scenes = _resolve_scenes(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_train: list[str] = []
    merged_test: list[str] = []
    kept_scene_paths: list[str] = []
    per_scene_meta: list[dict] = []
    errors: list[str] = []

    scene_keys_seen: set[str] = set()
    for scene in scenes:
        scene_dir = Path(scene)
        scene_key = derive_scene_key(scene_dir)
        if scene_key in scene_keys_seen:
            errors.append(f"Duplicate scene key '{scene_key}'")
            continue
        scene_keys_seen.add(scene_key)

        try:
            quality_meta = None
            if args.pose_quality_filter:
                quality_ok, quality_meta = _pose_quality(scene_dir, args)
                if not quality_ok:
                    msg = (
                        f"{scene_dir}: pose_quality_reject:{quality_meta['reason']} "
                        f"(pose_valid={quality_meta['pose_valid_images']}, "
                        f"coverage={quality_meta['coverage']:.3f}, "
                        f"covered_targets={quality_meta['covered_targets']})"
                    )
                    errors.append(msg)
                    continue

            names = _valid_image_names(scene_dir)
            train_names, test_names = _split_scene_names(
                names,
                test_ratio=args.test_ratio,
                train_ratio=args.train_ratio,
                seed=_stable_seed(scene_key, args.seed),
                min_train_images=args.min_train_images,
            )
        except Exception as e:
            errors.append(f"{scene_dir}: {e}")
            continue

        kept_scene_paths.append(str(scene_dir))
        merged_train.extend([scene_pair_key(scene_key, n) for n in train_names])
        merged_test.extend([scene_pair_key(scene_key, n) for n in test_names])

        per_scene_dir = output_dir / "per_scene" / scene_key.replace("/", "__")
        _write_names(per_scene_dir / "train_images.txt", train_names)
        _write_names(per_scene_dir / "test_images.txt", test_names)
        (per_scene_dir / "scene_path.txt").write_text(str(scene_dir) + "\n", encoding="utf-8")

        per_scene_meta.append(
            {
                "scene_key": scene_key,
                "scene_path": str(scene_dir),
                "total_valid_images": len(names),
                "train_images": len(train_names),
                "test_images": len(test_names),
                "pose_quality": quality_meta,
            }
        )

    if args.strict and errors:
        raise RuntimeError("Invalid scenes:\n" + "\n".join(errors))

    if len(kept_scene_paths) < args.min_scenes:
        detail = "\n".join(errors[:20])
        raise RuntimeError(
            f"Only {len(kept_scene_paths)} valid scenes after filtering; need at least {args.min_scenes}.\n"
            f"First errors:\n{detail}"
        )

    merged_train.sort()
    merged_test.sort()
    overlap = set(merged_train).intersection(merged_test)
    if overlap:
        raise RuntimeError(f"Split bug: {len(overlap)} scene-qualified images overlap between train/test")

    train_path = output_dir / "train_images.txt"
    test_path = output_dir / "test_images.txt"
    scenes_path = output_dir / "scenes.txt"
    _write_names(train_path, merged_train)
    _write_names(test_path, merged_test)
    _write_names(scenes_path, kept_scene_paths)

    meta = {
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "train_ratio": args.train_ratio,
        "requested_scenes": len(scenes),
        "valid_scenes": len(kept_scene_paths),
        "train_images": len(merged_train),
        "test_images": len(merged_test),
        "train_images_file": str(train_path),
        "test_images_file": str(test_path),
        "scenes_file": str(scenes_path),
        "per_scene": per_scene_meta,
        "errors": errors,
    }
    (output_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
