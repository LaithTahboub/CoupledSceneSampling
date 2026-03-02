"""Simulate dataset creation from scripts/train.sh defaults.

This utility:
1. Parses key default variables from train.sh.
2. Reproduces the train/test scene split logic.
3. Builds train/test MegaScenesDataset objects with those parameters.
4. Writes a JSON/Markdown report with dataset statistics.
5. Saves per-scene preview sheets with 5 triplets stacked vertically.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from css.data.dataset import MegaScenesDataset, load_image_tensor


def _read_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_shell_defaults(train_sh_path: Path) -> dict[str, str]:
    text = train_sh_path.read_text(encoding="utf-8")
    # Match patterns like: VAR=${VAR:-default_value}
    pattern = re.compile(r"^([A-Z0-9_]+)=\$\{\1:-([^}]*)\}", re.MULTILINE)
    defaults: dict[str, str] = {}
    for var, raw_value in pattern.findall(text):
        defaults[var] = _strip_quotes(raw_value.strip())
    return defaults


def _resolve_template(value: str, variables: dict[str, str]) -> str:
    def replace_braced(match: re.Match[str]) -> str:
        key = match.group(1)
        return variables.get(key, "")

    def replace_plain(match: re.Match[str]) -> str:
        key = match.group(1)
        return variables.get(key, "")

    out = value
    for _ in range(8):
        prev = out
        out = re.sub(r"\$\{([A-Z0-9_]+)\}", replace_braced, out)
        out = re.sub(r"\$([A-Z0-9_]+)", replace_plain, out)
        if out == prev:
            break
    return out


def _to_int(value: str | None, default: int | None = None) -> int | None:
    if value is None:
        return default
    v = value.strip()
    if v == "":
        return default
    return int(float(v))


def _to_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return float(default)
    return float(value)


def _sanitize_filename(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return s or "scene"


def _split_scenes(
    scenes: list[str],
    *,
    test_ratio: float,
    seed: int,
    min_train_scenes: int,
    max_scenes: int | None,
) -> tuple[list[str], list[str], list[str]]:
    selected = list(scenes)
    if max_scenes is not None and len(selected) > max_scenes:
        rng_sub = np.random.default_rng(seed)
        keep = rng_sub.choice(len(selected), size=max_scenes, replace=False)
        keep.sort()
        selected = [selected[i] for i in keep.tolist()]

    n = len(selected)
    if n < min_train_scenes + 1:
        raise ValueError(
            f"Need at least {min_train_scenes + 1} scenes after filtering; found {n}",
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_ratio))
    if test_ratio > 0 and n_test == 0:
        n_test = 1
    n_test = min(n_test, n - min_train_scenes)

    test_idx = set(perm[:n_test].tolist())
    train = [s for i, s in enumerate(selected) if i not in test_idx]
    test = [s for i, s in enumerate(selected) if i in test_idx]
    return selected, train, test


def _to_uint8(image_tensor) -> np.ndarray:
    return ((image_tensor.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()


def _render_scene_preview(
    scene_key: str,
    triplets: list[tuple],
    output_path: Path,
    *,
    H: int,
    W: int,
    rows: int = 5,
) -> None:
    canvas = Image.new("RGB", (3 * W, rows * H), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for row in range(rows):
        y0 = row * H
        if row < len(triplets):
            (
                _scene_name,
                images_dir,
                ref1_name,
                ref2_name,
                tgt_name,
                *_,
            ) = triplets[row]
            try:
                ref1_img, _, _ = load_image_tensor(images_dir, ref1_name, H, W)
                ref2_img, _, _ = load_image_tensor(images_dir, ref2_name, H, W)
                tgt_img, _, _ = load_image_tensor(images_dir, tgt_name, H, W)
                row_img = np.concatenate(
                    [_to_uint8(ref1_img), _to_uint8(ref2_img), _to_uint8(tgt_img)],
                    axis=1,
                )
                canvas.paste(Image.fromarray(row_img), (0, y0))
                label = f"{row + 1}: r1={Path(ref1_name).name} r2={Path(ref2_name).name} t={Path(tgt_name).name}"
            except Exception as exc:
                draw.rectangle([(0, y0), (3 * W - 1, y0 + H - 1)], fill=(70, 20, 20))
                label = f"{row + 1}: image load error ({exc})"
        else:
            draw.rectangle([(0, y0), (3 * W - 1, y0 + H - 1)], fill=(35, 35, 35))
            label = f"{row + 1}: no triplet"

        draw.rectangle([(0, y0), (3 * W - 1, y0 + 21)], fill=(0, 0, 0))
        draw.text((6, y0 + 4), label, fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _dataset_stats(dataset: MegaScenesDataset, split_scenes: list[str]) -> dict[str, object]:
    unique_images: set[str] = set()
    per_scene_triplets: dict[str, int] = defaultdict(int)
    for scene_name, images_dir, ref1_name, ref2_name, tgt_name, *_ in dataset.triplets:
        per_scene_triplets[scene_name] += 1
        unique_images.add(str(images_dir / ref1_name))
        unique_images.add(str(images_dir / ref2_name))
        unique_images.add(str(images_dir / tgt_name))

    split_scene_keys = [Path(s).name for s in split_scenes]
    missing = [s for s in split_scene_keys if per_scene_triplets.get(s, 0) == 0]
    values = list(per_scene_triplets.values())
    return {
        "num_scenes_in_split": len(split_scenes),
        "num_scenes_with_triplets": len(per_scene_triplets),
        "num_scenes_without_triplets": len(missing),
        "scenes_without_triplets": sorted(missing),
        "num_triplets": len(dataset.triplets),
        "num_unique_images": len(unique_images),
        "triplets_per_scene_min": int(min(values)) if values else 0,
        "triplets_per_scene_max": int(max(values)) if values else 0,
        "triplets_per_scene_mean": float(np.mean(values)) if values else 0.0,
        "triplets_per_scene_median": float(np.median(values)) if values else 0.0,
        "triplets_per_scene": dict(sorted(per_scene_triplets.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sh", type=str, default="scripts/train.sh")
    parser.add_argument("--output-dir", type=str, default="reports/dataset_sim")

    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--scenes-file", type=str, default=None)

    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--min-train-scenes", type=int, default=None)

    parser.add_argument("--max-pair-dist", type=float, default=None)
    parser.add_argument("--min-pair-iou", type=float, default=None)
    parser.add_argument("--min-ref-spacing", type=float, default=None)
    parser.add_argument("--min-view-cos", type=float, default=None)
    parser.add_argument("--max-rotation-deg", type=float, default=None)
    parser.add_argument("--max-focal-ratio", type=float, default=None)
    parser.add_argument("--pair-prefilter-topk", type=int, default=None)
    parser.add_argument("--candidate-pool-topk", type=int, default=None)
    parser.add_argument("--max-triplets", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--W", type=int, default=None)

    parser.add_argument("--preview-triplets", type=int, default=5)
    args = parser.parse_args()

    train_sh_path = Path(args.train_sh).resolve()
    defaults = _parse_shell_defaults(train_sh_path)

    root_default = defaults.get("ROOT", str(Path.cwd()))
    root = Path(args.root if args.root is not None else root_default).resolve()

    variables = dict(defaults)
    variables["ROOT"] = str(root)
    megascenes_root = _resolve_template(defaults.get("MEGASCENES_ROOT", "$ROOT/MegaScenes"), variables)
    variables["MEGASCENES_ROOT"] = megascenes_root

    scenes_file_str = args.scenes_file or _resolve_template(
        defaults.get("SCENES_FILE", "$MEGASCENES_ROOT/scenes_colmap_ready.txt"),
        variables,
    )
    scenes_file = Path(scenes_file_str).expanduser().resolve()
    if not scenes_file.exists():
        raise FileNotFoundError(f"Scenes file not found: {scenes_file}")

    test_ratio = args.test_ratio if args.test_ratio is not None else _to_float(defaults.get("TEST_RATIO"), 0.10)
    seed = args.seed if args.seed is not None else int(_to_float(defaults.get("SEED"), 42))
    max_scenes = args.max_scenes if args.max_scenes is not None else _to_int(defaults.get("MAX_SCENES"), None)
    min_train_scenes = (
        args.min_train_scenes
        if args.min_train_scenes is not None
        else _to_int(defaults.get("MIN_TRAIN_SCENES"), 1)
    )

    max_pair_dist = (
        args.max_pair_dist
        if args.max_pair_dist is not None
        else _to_float(defaults.get("MAX_PAIR_DIST"), 2.0)
    )
    min_pair_iou = (
        args.min_pair_iou
        if args.min_pair_iou is not None
        else _to_float(defaults.get("MIN_PAIR_IOU"), 0.22)
    )
    min_ref_spacing = (
        args.min_ref_spacing
        if args.min_ref_spacing is not None
        else _to_float(defaults.get("MIN_REF_SPACING"), 0.35)
    )
    min_view_cos = (
        args.min_view_cos
        if args.min_view_cos is not None
        else _to_float(defaults.get("MIN_VIEW_COS"), 0.90)
    )
    max_rotation_deg = (
        args.max_rotation_deg
        if args.max_rotation_deg is not None
        else _to_float(defaults.get("MAX_ROTATION_DEG"), 35.0)
    )
    max_focal_ratio = (
        args.max_focal_ratio
        if args.max_focal_ratio is not None
        else _to_float(defaults.get("MAX_FOCAL_RATIO"), 1.35)
    )
    pair_prefilter_topk = (
        args.pair_prefilter_topk
        if args.pair_prefilter_topk is not None
        else _to_int(defaults.get("PAIR_PREFILTER_TOPK"), 48)
    )
    candidate_pool_topk = (
        args.candidate_pool_topk
        if args.candidate_pool_topk is not None
        else _to_int(defaults.get("CANDIDATE_POOL_TOPK"), 20)
    )
    max_triplets = args.max_triplets if args.max_triplets is not None else _to_int(defaults.get("MAX_TRIPLETS"), 24)
    H = args.H if args.H is not None else _to_int(defaults.get("H"), 512)
    W = args.W if args.W is not None else _to_int(defaults.get("W"), 512)
    prompt_template = _strip_quotes(defaults.get("PROMPT_TEMPLATE", ""))
    prompt_template = prompt_template if prompt_template else None

    all_scenes = _read_lines(scenes_file)
    selected_scenes, train_scenes, test_scenes = _split_scenes(
        all_scenes,
        test_ratio=test_ratio,
        seed=seed,
        min_train_scenes=int(min_train_scenes),
        max_scenes=max_scenes,
    )

    print(f"[sim] scenes file: {scenes_file}")
    print(f"[sim] selected scenes: {len(selected_scenes)} | train: {len(train_scenes)} | test: {len(test_scenes)}")
    print(
        "[sim] dataset params: "
        f"max_pair_dist={max_pair_dist} min_pair_iou={min_pair_iou} "
        f"min_ref_spacing={min_ref_spacing} min_view_cos={min_view_cos} "
        f"max_rot_deg={max_rotation_deg} max_focal_ratio={max_focal_ratio} "
        f"prefilter_topk={pair_prefilter_topk} pool_topk={candidate_pool_topk} "
        f"max_triplets={max_triplets} H={H} W={W}",
    )

    train_dataset = MegaScenesDataset(
        train_scenes,
        H=int(H),
        W=int(W),
        max_pair_distance=float(max_pair_dist),
        max_triplets_per_scene=int(max_triplets),
        min_pair_iou=float(min_pair_iou),
        min_ref_spacing=float(min_ref_spacing),
        min_view_cos=float(min_view_cos),
        max_rotation_deg=float(max_rotation_deg),
        max_focal_ratio=float(max_focal_ratio),
        pair_prefilter_topk=int(pair_prefilter_topk),
        candidate_pool_topk=int(candidate_pool_topk),
        prompt_template=prompt_template,
    )
    test_dataset = MegaScenesDataset(
        test_scenes,
        H=int(H),
        W=int(W),
        max_pair_distance=float(max_pair_dist),
        max_triplets_per_scene=int(max_triplets),
        min_pair_iou=float(min_pair_iou),
        min_ref_spacing=float(min_ref_spacing),
        min_view_cos=float(min_view_cos),
        max_rotation_deg=float(max_rotation_deg),
        max_focal_ratio=float(max_focal_ratio),
        pair_prefilter_topk=int(pair_prefilter_topk),
        candidate_pool_topk=int(candidate_pool_topk),
        prompt_template=prompt_template,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    preview_dir = output_dir / "scene_triplet_previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    train_triplets_by_scene: dict[str, list[tuple]] = defaultdict(list)
    for triplet in train_dataset.triplets:
        train_triplets_by_scene[str(triplet[0])].append(triplet)

    for scene_path in train_scenes:
        scene_key = Path(scene_path).name
        scene_triplets = train_triplets_by_scene.get(scene_key, [])
        out_name = f"{_sanitize_filename(scene_key)}.png"
        _render_scene_preview(
            scene_key,
            scene_triplets[: max(1, int(args.preview_triplets))],
            preview_dir / out_name,
            H=int(H),
            W=int(W),
            rows=max(1, int(args.preview_triplets)),
        )

    report = {
        "train_sh": str(train_sh_path),
        "scenes_file": str(scenes_file),
        "parameters": {
            "test_ratio": float(test_ratio),
            "seed": int(seed),
            "max_scenes": (None if max_scenes is None else int(max_scenes)),
            "min_train_scenes": int(min_train_scenes),
            "max_pair_dist": float(max_pair_dist),
            "min_pair_iou": float(min_pair_iou),
            "min_ref_spacing": float(min_ref_spacing),
            "min_view_cos": float(min_view_cos),
            "max_rotation_deg": float(max_rotation_deg),
            "max_focal_ratio": float(max_focal_ratio),
            "pair_prefilter_topk": int(pair_prefilter_topk),
            "candidate_pool_topk": int(candidate_pool_topk),
            "max_triplets": int(max_triplets),
            "H": int(H),
            "W": int(W),
            "prompt_template": prompt_template,
        },
        "split": {
            "all_scenes_count": len(all_scenes),
            "selected_scenes_count": len(selected_scenes),
            "train_scenes_count": len(train_scenes),
            "test_scenes_count": len(test_scenes),
            "train_scenes": train_scenes,
            "test_scenes": test_scenes,
        },
        "train_dataset": _dataset_stats(train_dataset, train_scenes),
        "test_dataset": _dataset_stats(test_dataset, test_scenes),
        "preview_dir": str(preview_dir),
    }

    report_json_path = output_dir / "dataset_report.json"
    report_md_path = output_dir / "dataset_report.md"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Dataset Simulation Report",
        "",
        f"- train.sh: `{train_sh_path}`",
        f"- scenes file: `{scenes_file}`",
        f"- selected scenes: {len(selected_scenes)}",
        f"- train scenes: {len(train_scenes)}",
        f"- test scenes: {len(test_scenes)}",
        f"- train triplets: {report['train_dataset']['num_triplets']}",
        f"- train unique images: {report['train_dataset']['num_unique_images']}",
        f"- test triplets: {report['test_dataset']['num_triplets']}",
        f"- test unique images: {report['test_dataset']['num_unique_images']}",
        f"- preview directory: `{preview_dir}`",
        "",
        "See `dataset_report.json` for full details (including per-scene triplet counts).",
    ]
    report_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[sim] wrote report: {report_json_path}")
    print(f"[sim] wrote summary: {report_md_path}")
    print(f"[sim] wrote previews: {preview_dir}")


if __name__ == "__main__":
    main()
