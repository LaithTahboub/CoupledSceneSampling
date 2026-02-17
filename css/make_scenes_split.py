"""Create deterministic train/test splits from a scenes list file."""

import argparse
import json
from pathlib import Path

import numpy as np


def _read_scenes(path: Path) -> list[str]:
    scenes: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        scenes.append(line)
    return scenes


def _write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes-file", required=True, help="Input list of scene directories")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-train-scenes", type=int, default=1)
    args = p.parse_args()

    if not (0.0 <= args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be in [0, 1)")

    scenes = _read_scenes(Path(args.scenes_file))
    n = len(scenes)
    if n < args.min_train_scenes + 1:
        raise ValueError(f"Need at least {args.min_train_scenes + 1} scenes, found {n}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)

    n_test = int(round(n * args.test_ratio))
    if args.test_ratio > 0 and n_test == 0:
        n_test = 1
    n_test = min(n_test, n - args.min_train_scenes)

    test_idx = set(perm[:n_test].tolist())
    train_scenes = [s for i, s in enumerate(scenes) if i not in test_idx]
    test_scenes = [s for i, s in enumerate(scenes) if i in test_idx]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_scenes.txt"
    test_path = output_dir / "test_scenes.txt"
    _write_lines(train_path, train_scenes)
    _write_lines(test_path, test_scenes)

    meta = {
        "scenes_file": str(Path(args.scenes_file)),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "total_scenes": n,
        "train_scenes": len(train_scenes),
        "test_scenes": len(test_scenes),
        "train_scenes_file": str(train_path),
        "test_scenes_file": str(test_path),
    }
    (output_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
