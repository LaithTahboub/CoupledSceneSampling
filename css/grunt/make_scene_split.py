"""Create deterministic train/test image splits for a scene."""

import argparse
import json
from pathlib import Path

import numpy as np

from css.data.colmap_reader import read_scene


def _valid_image_names(scene_dir: Path) -> list[str]:
    _, images = read_scene(scene_dir)
    images_dir = scene_dir / "images"
    names = [img.name for img in images.values() if (images_dir / img.name).exists()]
    return sorted(set(names))


def _write_names(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="Scene path (e.g., MegaScenes/Mysore_Palace)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--train-ratio", type=float, default=1.0, help="Fraction of non-test images kept for training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-train-images", type=int, default=3)
    args = p.parse_args()

    scene_dir = Path(args.scene)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = _valid_image_names(scene_dir)
    n = len(names)
    if n < args.min_train_images + 1:
        raise ValueError(f"Need at least {args.min_train_images + 1} valid images, found {n}")

    if not (0.0 <= args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be in [0, 1)")
    if not (0.0 < args.train_ratio <= 1.0):
        raise ValueError("--train-ratio must be in (0, 1]")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)

    n_test = int(round(n * args.test_ratio))
    if args.test_ratio > 0 and n_test == 0:
        n_test = 1
    n_test = min(n_test, n - args.min_train_images)

    test_idx = set(perm[:n_test].tolist())
    train_names_all = [name for i, name in enumerate(names) if i not in test_idx]
    test_names = [name for i, name in enumerate(names) if i in test_idx]

    n_train_keep = int(round(len(train_names_all) * args.train_ratio))
    n_train_keep = max(args.min_train_images, min(len(train_names_all), n_train_keep))
    train_perm = rng.permutation(len(train_names_all))
    keep_idx = set(train_perm[:n_train_keep].tolist())
    train_names = [name for i, name in enumerate(train_names_all) if i in keep_idx]

    train_names.sort()
    test_names.sort()
    overlap = set(train_names).intersection(test_names)
    if overlap:
        raise RuntimeError(f"Split bug: {len(overlap)} images overlap between train/test")

    train_path = output_dir / "train_images.txt"
    test_path = output_dir / "test_images.txt"
    _write_names(train_path, train_names)
    _write_names(test_path, test_names)

    meta = {
        "scene": str(scene_dir),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "train_ratio": args.train_ratio,
        "total_valid_images": n,
        "train_images": len(train_names),
        "test_images": len(test_names),
        "train_images_file": str(train_path),
        "test_images_file": str(test_path),
    }
    (output_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
