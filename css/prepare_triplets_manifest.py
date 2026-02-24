"""prepare packed best-k triplets by downloading one scene at a time."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from css.data.dataset import MegaScenesDataset


def _to_scene_id(value):
    if isinstance(value, dict):
        for key in ("id", "scene_id", "sid", "index"):
            if key in value:
                return _to_scene_id(value[key])
        return None
    if isinstance(value, list):
        for item in value:
            sid = _to_scene_id(item)
            if sid is not None:
                return sid
        return None
    if isinstance(value, int):
        s = f"{value:06d}"
        return f"{s[:3]}/{s[3:]}"
    if isinstance(value, str):
        s = value.strip()
        if "/" in s:
            return s
        if s.isdigit():
            s = s.zfill(6)
            return f"{s[:3]}/{s[3:]}"
    return None


def _sanitize_scene_name(name: str) -> str:
    s = name.replace("/", "__")
    s = s.replace('"', "").replace("'", "")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._")
    return s or "scene"


def _is_scene_id(token: str) -> bool:
    return bool(re.fullmatch(r"\d{3}/\d{3}", token.strip()))


def _build_candidates(categories_json: Path, scene_list_file: Path | None) -> list[tuple[str, str, str]]:
    categories = json.loads(categories_json.read_text(encoding="utf-8"))
    if not isinstance(categories, dict):
        raise ValueError(f"Unexpected categories format in {categories_json}")

    id_to_name: dict[str, str] = {}
    for raw_name, raw_value in categories.items():
        sid = _to_scene_id(raw_value)
        if sid is None:
            continue
        id_to_name.setdefault(sid, raw_name)

    selected: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    if scene_list_file is not None:
        lines = scene_list_file.read_text(encoding="utf-8").splitlines()
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            sid = None
            raw_name = None
            if "\t" in raw:
                left, right = raw.split("\t", 1)
                if _is_scene_id(left):
                    sid = left.strip()
                    raw_name = right.strip() or id_to_name.get(sid, sid.replace("/", "_"))
            elif _is_scene_id(raw):
                sid = raw
                raw_name = id_to_name.get(sid, sid.replace("/", "_"))
            else:
                for name in (raw, raw.replace(" ", "_"), raw.replace("_", " ")):
                    if name in categories:
                        sid = _to_scene_id(categories[name])
                        raw_name = name
                        break

            if sid is None or sid in seen:
                continue
            seen.add(sid)
            safe_name = f"{sid.replace('/', '_')}__{_sanitize_scene_name(raw_name or sid)}"
            selected.append((sid, safe_name, raw_name or sid))
        return selected

    for sid, raw_name in id_to_name.items():
        if sid in seen:
            continue
        seen.add(sid)
        safe_name = f"{sid.replace('/', '_')}__{_sanitize_scene_name(raw_name)}"
        selected.append((sid, safe_name, raw_name))
    return selected


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def _run_download(
    download_script: Path,
    scene_id: str,
    scene_dir: Path,
    scene_name: str,
    download_sparse: bool,
    recon_no: int,
) -> bool:
    env = os.environ.copy()
    env["DOWNLOAD_SPARSE"] = "1" if download_sparse else "0"
    env["RECON_NO"] = str(recon_no)
    cmd = ["bash", str(download_script), scene_id, str(scene_dir), scene_name]
    proc = subprocess.run(cmd, env=env)
    return proc.returncode == 0


def _read_manifest_stats(manifest_path: Path) -> tuple[set[str], int]:
    if not manifest_path.exists():
        return set(), 0
    done: set[str] = set()
    rows = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            rows += 1
            sid = str(row.get("scene_id", "")).strip()
            if sid:
                done.add(sid)
    return done, rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True, help="Root directory for packed triplets and manifest")
    p.add_argument("--categories-json", required=True, help="Path to categories.json")
    p.add_argument("--scene-list-file", type=str, default=None, help="Optional candidate scene subset")
    p.add_argument("--download-script", default="scripts/grunt/download_scene_data.sh")
    p.add_argument("--target-scenes", type=int, default=100, help="How many scenes with >=1 triplet to keep")
    p.add_argument("--max-candidates", type=int, default=None, help="Optional cap before processing")
    p.add_argument("--max-triplets-per-scene", type=int, default=3)
    p.add_argument("--max-pair-dist", type=float, default=2.5)
    p.add_argument("--min-dir-sim", type=float, default=0.2)
    p.add_argument("--min-ref-spacing", type=float, default=0.25)
    p.add_argument("--min-images-per-scene", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download-sparse", action="store_true", default=True)
    p.add_argument("--no-download-sparse", action="store_false", dest="download_sparse")
    p.add_argument("--recon-no", type=int, default=0)
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--keep-temp-scenes", action="store_true")
    args = p.parse_args()

    if args.target_scenes <= 0:
        raise ValueError("--target-scenes must be >= 1")
    if args.max_triplets_per_scene <= 0:
        raise ValueError("--max-triplets-per-scene must be >= 1")

    output_root = Path(args.output_root).resolve()
    packed_scenes_root = output_root / "scenes"
    tmp_root = output_root / "tmp_scenes"
    manifest_path = output_root / "triplets_manifest.jsonl"
    packed_scenes_file = output_root / "packed_scenes.txt"
    prep_meta_path = output_root / "prep_meta.json"
    categories_json = Path(args.categories_json).resolve()
    scene_list_file = Path(args.scene_list_file).resolve() if args.scene_list_file is not None else None
    download_script = Path(args.download_script).resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    packed_scenes_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    candidates = _build_candidates(categories_json, scene_list_file)
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if len(candidates) == 0:
        raise ValueError("No candidate scenes found")

    import numpy as np

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(candidates))
    candidates = [candidates[i] for i in perm.tolist()]

    done_scene_ids, existing_rows = _read_manifest_stats(manifest_path) if args.resume else (set(), 0)
    if not args.resume:
        manifest_path.write_text("", encoding="utf-8")
        packed_scenes_file.write_text("", encoding="utf-8")

    kept_scene_ids: set[str] = set(done_scene_ids)
    kept_scene_count = len(done_scene_ids)
    kept_triplet_count = existing_rows
    processed = 0
    skipped = 0

    manifest_mode = "a" if args.resume else "w"
    with open(manifest_path, manifest_mode, encoding="utf-8") as mf, open(
        packed_scenes_file, "a" if args.resume else "w", encoding="utf-8"
    ) as sf:
        for scene_id, safe_name, raw_name in candidates:
            if kept_scene_count >= args.target_scenes:
                break
            if scene_id in kept_scene_ids:
                continue

            processed += 1
            tmp_scene_dir = tmp_root / safe_name
            if tmp_scene_dir.exists():
                shutil.rmtree(tmp_scene_dir)

            print(f"[prep] downloading {scene_id} ({raw_name})")
            ok = _run_download(
                download_script=download_script,
                scene_id=scene_id,
                scene_dir=tmp_scene_dir,
                scene_name=raw_name,
                download_sparse=args.download_sparse,
                recon_no=args.recon_no,
            )
            if not ok:
                skipped += 1
                if tmp_scene_dir.exists():
                    shutil.rmtree(tmp_scene_dir, ignore_errors=True)
                continue

            image_count = _count_files(tmp_scene_dir / "images")
            if image_count < args.min_images_per_scene:
                print(f"[prep] skip {scene_id}: images={image_count} < {args.min_images_per_scene}")
                skipped += 1
                if not args.keep_temp_scenes:
                    shutil.rmtree(tmp_scene_dir, ignore_errors=True)
                continue

            scene_dataset = MegaScenesDataset(
                [str(tmp_scene_dir)],
                H=args.H,
                W=args.W,
                max_pair_distance=args.max_pair_dist,
                max_triplets_per_scene=args.max_triplets_per_scene,
                min_dir_similarity=args.min_dir_sim,
                min_ref_spacing=args.min_ref_spacing,
            )
            triplets = list(scene_dataset.triplets)
            if len(triplets) == 0:
                print(f"[prep] skip {scene_id}: no valid triplets")
                skipped += 1
                if not args.keep_temp_scenes:
                    shutil.rmtree(tmp_scene_dir, ignore_errors=True)
                continue

            packed_scene_dir = packed_scenes_root / safe_name
            packed_images_dir = packed_scene_dir / "images"
            if packed_scene_dir.exists():
                shutil.rmtree(packed_scene_dir)
            packed_images_dir.mkdir(parents=True, exist_ok=True)

            image_names = set()
            for _, _, ref1_name, ref2_name, target_name, *_ in triplets:
                image_names.add(ref1_name)
                image_names.add(ref2_name)
                image_names.add(target_name)

            src_images_dir = tmp_scene_dir / "images"
            for image_name in sorted(image_names):
                src = src_images_dir / image_name
                dst = packed_images_dir / image_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

            (packed_scene_dir / "scene_name_raw.txt").write_text(raw_name + "\n", encoding="utf-8")
            (packed_scene_dir / "scene_id.txt").write_text(scene_id + "\n", encoding="utf-8")

            images_dir_rel = packed_images_dir.relative_to(output_root).as_posix()
            for rank, triplet in enumerate(triplets):
                (
                    scene_name,
                    _images_dir,
                    ref1_name,
                    ref2_name,
                    target_name,
                    ref1_c2w,
                    ref2_c2w,
                    target_c2w,
                    K_ref1,
                    K_ref2,
                    K_tgt,
                ) = triplet
                row = {
                    "scene_id": scene_id,
                    "scene_name": scene_name,
                    "scene_prompt_name": raw_name,
                    "images_dir": images_dir_rel,
                    "ref1_name": ref1_name,
                    "ref2_name": ref2_name,
                    "target_name": target_name,
                    "ref1_c2w": ref1_c2w.tolist(),
                    "ref2_c2w": ref2_c2w.tolist(),
                    "target_c2w": target_c2w.tolist(),
                    "K_ref1": K_ref1.tolist(),
                    "K_ref2": K_ref2.tolist(),
                    "K_tgt": K_tgt.tolist(),
                    "triplet_rank": rank,
                }
                mf.write(json.dumps(row) + "\n")

            sf.write(str(packed_scene_dir) + "\n")
            kept_scene_ids.add(scene_id)
            kept_scene_count += 1
            kept_triplet_count += len(triplets)
            print(
                f"[prep] kept scene {scene_id}: triplets={len(triplets)} "
                f"unique_images={len(image_names)}"
            )

            if not args.keep_temp_scenes:
                shutil.rmtree(tmp_scene_dir, ignore_errors=True)

    meta = {
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "packed_scenes_file": str(packed_scenes_file),
        "categories_json": str(categories_json),
        "scene_list_file": str(scene_list_file) if scene_list_file is not None else None,
        "target_scenes": args.target_scenes,
        "max_triplets_per_scene": args.max_triplets_per_scene,
        "max_pair_dist": args.max_pair_dist,
        "min_dir_sim": args.min_dir_sim,
        "min_ref_spacing": args.min_ref_spacing,
        "min_images_per_scene": args.min_images_per_scene,
        "seed": args.seed,
        "processed_candidates": processed,
        "kept_scenes": kept_scene_count,
        "kept_triplets": kept_triplet_count,
        "skipped": skipped,
    }
    prep_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()