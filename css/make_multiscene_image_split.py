"""Create deterministic per-scene image train/test splits for many scenes."""

import argparse
import json
from pathlib import Path

import numpy as np

from css.data.colmap_reader import read_scene


def _read_lines(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def _resolve_scene_path(scene_spec: str, scenes_root: Path | None) -> Path:
    path = Path(scene_spec)
    if path.is_absolute() or scenes_root is None:
        return path
    return scenes_root / path


def _valid_image_names(scene_dir: Path) -> list[str]:
    _, images = read_scene(scene_dir)
    images_dir = scene_dir / "images"
    names = [img.name for img in images.values() if (images_dir / img.name).exists()]
    return sorted(set(names))


def _split_scene(
    names: list[str],
    rng: np.random.Generator,
    test_ratio: float,
    train_ratio: float,
    min_train_images: int,
) -> tuple[list[str], list[str]]:
    n = len(names)
    n_test = int(round(n * test_ratio))
    if test_ratio > 0 and n_test == 0:
        n_test = 1
    n_test = min(n_test, n - min_train_images)

    perm = rng.permutation(n)
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
    return train_names, test_names


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes-file", required=True, help="Input list of scene directories")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--scenes-root", default=None, help="Optional base path for relative scene entries")
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--train-ratio", type=float, default=1.0, help="Fraction of non-test images kept for training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-train-images", type=int, default=3)
    args = p.parse_args()

    if not (0.0 <= args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be in [0, 1)")
    if not (0.0 < args.train_ratio <= 1.0):
        raise ValueError("--train-ratio must be in (0, 1]")

    scenes_root = Path(args.scenes_root) if args.scenes_root is not None else None
    scene_specs = _read_lines(Path(args.scenes_file))
    if len(scene_specs) == 0:
        raise ValueError(f"No scenes found in {args.scenes_file}")

    rng = np.random.default_rng(args.seed)
    train_keys: list[str] = []
    test_keys: list[str] = []
    skipped: list[tuple[str, str]] = []
    per_scene: list[dict] = []

    for scene_spec in scene_specs:
        scene_dir = _resolve_scene_path(scene_spec, scenes_root)
        scene_key = scene_dir.name
        if not scene_dir.exists():
            skipped.append((scene_spec, "scene directory missing"))
            continue

        try:
            names = _valid_image_names(scene_dir)
        except Exception as e:
            skipped.append((scene_spec, f"load failed: {e}"))
            continue

        if len(names) < args.min_train_images + 1:
            skipped.append((scene_spec, f"fewer than {args.min_train_images + 1} valid images"))
            continue

        train_names, test_names = _split_scene(
            names=names,
            rng=rng,
            test_ratio=args.test_ratio,
            train_ratio=args.train_ratio,
            min_train_images=args.min_train_images,
        )
        overlap = set(train_names).intersection(test_names)
        if overlap:
            raise RuntimeError(f"Split bug for {scene_key}: train/test image overlap")

        train_keys.extend([f"{scene_key}/{name}" for name in train_names])
        test_keys.extend([f"{scene_key}/{name}" for name in test_names])
        per_scene.append(
            {
                "scene_spec": scene_spec,
                "scene_name": scene_key,
                "total_valid_images": len(names),
                "train_images": len(train_names),
                "test_images": len(test_names),
            }
        )

    if len(train_keys) == 0:
        raise ValueError("No train images selected across scenes")
    if args.test_ratio > 0 and len(test_keys) == 0:
        raise ValueError("No test images selected across scenes")

    train_keys = sorted(set(train_keys))
    test_keys = sorted(set(test_keys))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_images.txt"
    test_path = output_dir / "test_images.txt"
    skipped_path = output_dir / "skipped_scenes.tsv"

    _write_lines(train_path, train_keys)
    _write_lines(test_path, test_keys)

    with open(skipped_path, "w", encoding="utf-8") as f:
        f.write("scene_spec\treason\n")
        for scene_spec, reason in skipped:
            f.write(f"{scene_spec}\t{reason}\n")

    meta = {
        "scenes_file": str(Path(args.scenes_file)),
        "scenes_root": str(scenes_root) if scenes_root is not None else None,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "train_ratio": args.train_ratio,
        "min_train_images": args.min_train_images,
        "requested_scenes": len(scene_specs),
        "split_scenes": len(per_scene),
        "skipped_scenes": len(skipped),
        "train_images": len(train_keys),
        "test_images": len(test_keys),
        "train_images_file": str(train_path),
        "test_images_file": str(test_path),
        "skipped_scenes_file": str(skipped_path),
        "per_scene": per_scene,
    }
    (output_dir / "split_images_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
