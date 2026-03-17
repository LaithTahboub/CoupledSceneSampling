"""Target-centric scene dataset with difficulty-bucketed sampling.

Builds (ref1, ref2, target) records from MegaScenes COLMAP reconstructions
with ref1-anchored Plucker maps. Supports multi-target training by grouping
records that share the same reference pair.

Mining algorithm (target-centric):
1. For each target, build a pool of valid reference candidates.
2. Form all valid (ref1, ref2) pairs from the pool; reject geometric duplicates.
3. Score pairs, sort, then greedily keep a small capped set of geometrically
   distinct pairs per target (default 6).
4. Allocate the per-scene budget by round-robin over targets so every target
   gets coverage before any target gets extra pairs.
5. Assign each kept record to exactly one difficulty bucket.
6. Bucket-ratio balancing happens at sampling time, not here.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from css.data.colmap_reader import read_scene
from css.data.dataset import (
    build_cropped_scaled_intrinsics,
    compute_plucker_tensor,
    compute_scene_scale,
    load_image_tensor,
    normalize_c2w_translation,
)
from css.data.iou import compute_covisibility


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class BucketConfig:
    """Covisibility and distance ranges for a difficulty bucket."""
    min_covis: float
    max_covis: float
    min_distance: float
    max_distance: float


@dataclass
class SceneRecord:
    scene_name: str
    images_dir: Path
    ref1_name: str
    ref2_name: str
    target_name: str
    prompt: str
    ref1_c2w: np.ndarray   # (4, 4)
    ref2_c2w: np.ndarray
    tgt_c2w: np.ndarray
    K_ref1: np.ndarray     # (3, 3) adjusted for crop+resize
    K_ref2: np.ndarray
    K_tgt: np.ndarray
    score: float
    scene_scale: float = 1.0  # from compute_scene_scale(); used to normalize translations
    difficulty: Difficulty = Difficulty.MEDIUM


# Keep old name as alias for backward compat
TripletRecord = SceneRecord


# -----------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------

def _viewing_direction(c2w: np.ndarray) -> np.ndarray:
    d = -c2w[:3, 2].astype(np.float64)
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else np.array([0.0, 0.0, -1.0])


def _focal_length(K: np.ndarray) -> float:
    return float((abs(K[0, 0]) + abs(K[1, 1])) / 2.0)


def _focal_ratio(K_a: np.ndarray, K_b: np.ndarray) -> float:
    fa, fb = _focal_length(K_a), _focal_length(K_b)
    if fa < 1e-6 or fb < 1e-6:
        return float("inf")
    return max(fa, fb) / min(fa, fb)


def _is_geometric_duplicate(
    covis: float, dist: float, orient_dot: float, *,
    covis_thresh: float = 0.90,
    dist_thresh: float = 0.02,
    orient_thresh: float = 0.995,
) -> bool:
    """Two cameras are geometric duplicates when ALL three hold."""
    return covis > covis_thresh and dist < dist_thresh and orient_dot > orient_thresh


def _score_triplet(
    cov_r1_t: float, cov_r2_t: float, cov_r1_r2: float,
    dist_r1_t: float, dist_r2_t: float,
) -> float:
    overlap = cov_r1_t + cov_r2_t
    redundancy = 0.3 * cov_r1_r2
    baseline = 0.1 * min(dist_r1_t + dist_r2_t, 1.0)
    return overlap - redundancy + baseline


def _assign_difficulty(
    avg_covis: float, avg_dist: float,
    bucket_configs: dict[Difficulty, BucketConfig],
) -> Difficulty | None:
    for diff in (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD):
        cfg = bucket_configs[diff]
        if (cfg.min_covis <= avg_covis <= cfg.max_covis
                and cfg.min_distance <= avg_dist <= cfg.max_distance):
            return diff
    return None


def _ref_pairs_are_similar(
    pair_a: tuple[int, int], pair_b: tuple[int, int],
    positions: dict[int, np.ndarray], *, pos_thresh: float = 0.03,
) -> bool:
    a0, a1 = pair_a
    b0, b1 = pair_b
    d00 = float(np.linalg.norm(positions[a0] - positions[b0]))
    d11 = float(np.linalg.norm(positions[a1] - positions[b1]))
    if d00 < pos_thresh and d11 < pos_thresh:
        return True
    d01 = float(np.linalg.norm(positions[a0] - positions[b1]))
    d10 = float(np.linalg.norm(positions[a1] - positions[b0]))
    return d01 < pos_thresh and d10 < pos_thresh


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class MegaScenesDataset(Dataset):
    """Target-centric scene dataset with difficulty buckets and multi-target support."""

    def __init__(
        self,
        scene_dirs: list[str],
        *,
        H: int = 256,
        W: int = 256,
        caption_dir: str | None = None,
        # Difficulty bucket ranges
        easy_min_covis: float = 0.45,  easy_max_covis: float = 0.80,
        easy_min_distance: float = 0.02, easy_max_distance: float = 0.15,
        medium_min_covis: float = 0.25, medium_max_covis: float = 0.50,
        medium_min_distance: float = 0.08, medium_max_distance: float = 0.40,
        hard_min_covis: float = 0.12,  hard_max_covis: float = 0.30,
        hard_min_distance: float = 0.15, hard_max_distance: float = 1.0,
        # Bucket sampling ratios (enforced at sampling time)
        easy_ratio: float = 0.50,
        medium_ratio: float = 0.35,
        hard_ratio: float = 0.15,
        # Reference-reference constraints
        min_ref_covisibility: float = 0.10,
        max_ref_covisibility: float = 0.70,
        # Quality filtering
        max_triplets_per_scene: int = 80,
        min_orientation_dot: float = 0.5,
        max_focal_length_ratio: float = 2.0,
        min_points_per_image: int = 400,
        reject_near_duplicate_refs: bool = True,
        near_duplicate_threshold: float = 0.90,
        # Per-target cap on kept ref-pairs
        max_pairs_per_target: int = 6,
        # Similarity threshold for dedup of kept pairs (camera-center distance)
        pair_similarity_thresh: float = 0.03,
        # Minimum unique targets required to keep a scene (ensures diversity)
        min_targets_per_scene: int = 1,
    ):
        self.H, self.W = H, W
        self.latent_h, self.latent_w = H // 8, W // 8

        self.bucket_configs = {
            Difficulty.EASY: BucketConfig(easy_min_covis, easy_max_covis,
                                         easy_min_distance, easy_max_distance),
            Difficulty.MEDIUM: BucketConfig(medium_min_covis, medium_max_covis,
                                           medium_min_distance, medium_max_distance),
            Difficulty.HARD: BucketConfig(hard_min_covis, hard_max_covis,
                                         hard_min_distance, hard_max_distance),
        }
        self.bucket_ratios = {
            Difficulty.EASY: easy_ratio,
            Difficulty.MEDIUM: medium_ratio,
            Difficulty.HARD: hard_ratio,
        }

        self.min_ref_covisibility = min_ref_covisibility
        self.max_ref_covisibility = max_ref_covisibility
        self.min_orientation_dot = min_orientation_dot
        self.max_focal_length_ratio = max_focal_length_ratio
        self.min_points_per_image = min_points_per_image
        self.reject_near_duplicate_refs = reject_near_duplicate_refs
        self.near_duplicate_threshold = near_duplicate_threshold
        self.max_pairs_per_target = max_pairs_per_target
        self.pair_similarity_thresh = pair_similarity_thresh

        # Load precomputed captions (per-scene JSON files from caption_dataset.py)
        self._captions: dict[str, dict[str, str]] = {}
        if caption_dir is not None:
            caption_path = Path(caption_dir)
            for scene_spec in scene_dirs:
                scene_name = Path(scene_spec).name
                cf = caption_path / f"{scene_name}.json"
                if cf.exists():
                    with open(cf) as f:
                        self._captions[scene_name] = json.load(f)

        # Mine
        self.records: list[SceneRecord] = []
        n_skipped_diversity = 0
        for scene_spec in scene_dirs:
            scene_records = self._mine_scene(Path(scene_spec), max_triplets_per_scene)
            if scene_records and min_targets_per_scene > 1:
                n_unique_targets = len({r.target_name for r in scene_records})
                if n_unique_targets < min_targets_per_scene:
                    n_skipped_diversity += 1
                    continue
            self.records.extend(scene_records)

        # Backward compat alias
        self.triplets = self.records

        self.bucket_counts = {d.value: 0 for d in Difficulty}
        for t in self.records:
            self.bucket_counts[t.difficulty.value] += 1

        if n_skipped_diversity:
            print(f"MegaScenesDataset: skipped {n_skipped_diversity} scenes "
                  f"with < {min_targets_per_scene} unique targets")

        if self.records:
            print(f"MegaScenesDataset: {len(self.records)} records "
                  f"from {len(scene_dirs) - n_skipped_diversity} scenes")
            for d in Difficulty:
                c = self.bucket_counts[d.value]
                pct = c / len(self.records) * 100
                print(f"  {d.value}: {c} ({pct:.1f}%)")
        else:
            print("MegaScenesDataset: 0 valid records found.")

    # ---- public helpers -----------------------------------------------

    def get_bucket_indices(self) -> dict[Difficulty, list[int]]:
        indices: dict[Difficulty, list[int]] = {d: [] for d in Difficulty}
        for i, t in enumerate(self.records):
            indices[t.difficulty].append(i)
        return indices

    # ---- target-centric mining ----------------------------------------

    def _mine_scene(
        self, scene_dir: Path, max_triplets: int,
    ) -> list[SceneRecord]:
        try:
            cameras, images = read_scene(scene_dir)
        except Exception as e:
            print(f"  SKIP {scene_dir.name}: failed to read COLMAP data: {e}")
            return []

        images_dir = scene_dir / "images"
        scene_name = scene_dir.name
        scene_captions = self._captions.get(scene_name, {})

        disk_files: set[str] = set()
        for p in images_dir.rglob("*"):
            if p.is_file():
                disk_files.add(p.relative_to(images_dir).as_posix())

        valid = []
        resolved: dict[int, str] = {}
        for img in images.values():
            name = img.name.replace("\\", "/")
            if name not in disk_files:
                continue
            n_pts = len(img.point3d_ids) if img.point3d_ids is not None else 0
            if n_pts < self.min_points_per_image:
                continue
            resolved[img.id] = name
            valid.append(img)

        if len(valid) < 3:
            return []

        positions = {img.id: img.c2w[:3, 3].astype(np.float64) for img in valid}
        view_dirs = {img.id: _viewing_direction(img.c2w) for img in valid}
        images_by_id = {img.id: img for img in valid}

        # Scene scale for translation normalization
        pos_array = np.stack(list(positions.values()))
        scene_scale = compute_scene_scale(pos_array, percentile=95.0)
        if scene_scale < 1e-4:
            print(f"  SKIP {scene_name}: degenerate scene scale ({scene_scale:.2e})")
            return []
        K_by_id = {
            img.id: build_cropped_scaled_intrinsics(
                cameras[img.camera_id], self.H, self.W
            ).astype(np.float32)
            for img in valid
        }

        covis_cache: dict[tuple[int, int], float] = {}
        dist_cache: dict[tuple[int, int], float] = {}
        orient_cache: dict[tuple[int, int], float] = {}

        def _pk(a: int, b: int) -> tuple[int, int]:
            return (a, b) if a < b else (b, a)

        def covis(a: int, b: int) -> float:
            k = _pk(a, b)
            if k not in covis_cache:
                covis_cache[k] = compute_covisibility(
                    images_by_id[k[0]], images_by_id[k[1]])
            return covis_cache[k]

        def dist(a: int, b: int) -> float:
            k = _pk(a, b)
            if k not in dist_cache:
                dist_cache[k] = float(
                    np.linalg.norm(positions[k[0]] - positions[k[1]]))
            return dist_cache[k]

        def orient(a: int, b: int) -> float:
            k = _pk(a, b)
            if k not in orient_cache:
                orient_cache[k] = float(
                    np.dot(view_dirs[k[0]], view_dirs[k[1]]))
            return orient_cache[k]

        min_covis_any = min(c.min_covis for c in self.bucket_configs.values())
        max_dist_any = max(c.max_distance for c in self.bucket_configs.values())

        per_target: dict[int, list[SceneRecord]] = {}

        for tgt in valid:
            refs: list[int] = []
            for ref in valid:
                if ref.id == tgt.id:
                    continue
                if covis(ref.id, tgt.id) < min_covis_any:
                    continue
                if dist(ref.id, tgt.id) > max_dist_any:
                    continue
                if orient(ref.id, tgt.id) < self.min_orientation_dot:
                    continue
                if _focal_ratio(K_by_id[ref.id], K_by_id[tgt.id]) > self.max_focal_length_ratio:
                    continue
                refs.append(ref.id)

            if len(refs) < 2:
                continue

            scored_pairs: list[tuple[float, int, int, Difficulty]] = []

            for ii in range(len(refs)):
                r1 = refs[ii]
                for jj in range(ii + 1, len(refs)):
                    r2 = refs[jj]

                    # Normalize ordering: smaller ID is always ref1.
                    # This ensures consistent Plucker anchoring across
                    # different targets sharing the same ref pair.
                    if r1 > r2:
                        r1, r2 = r2, r1

                    cv_rr = covis(r1, r2)
                    if cv_rr < self.min_ref_covisibility:
                        continue
                    if cv_rr > self.max_ref_covisibility:
                        continue

                    if self.reject_near_duplicate_refs:
                        if _is_geometric_duplicate(
                            cv_rr, dist(r1, r2), orient(r1, r2),
                            covis_thresh=self.near_duplicate_threshold,
                        ):
                            continue

                    if _focal_ratio(K_by_id[r1], K_by_id[r2]) > self.max_focal_length_ratio:
                        continue
                    if orient(r1, r2) < self.min_orientation_dot:
                        continue

                    cv1 = covis(r1, tgt.id)
                    cv2 = covis(r2, tgt.id)
                    d1 = dist(r1, tgt.id)
                    d2 = dist(r2, tgt.id)
                    sc = _score_triplet(cv1, cv2, cv_rr, d1, d2)

                    avg_cv = (cv1 + cv2) / 2.0
                    avg_d = (d1 + d2) / 2.0
                    diff = _assign_difficulty(avg_cv, avg_d, self.bucket_configs)
                    if diff is None:
                        continue

                    scored_pairs.append((sc, r1, r2, diff))

            if not scored_pairs:
                continue

            scored_pairs.sort(key=lambda x: x[0], reverse=True)

            kept_pairs: list[tuple[int, int]] = []
            kept_records: list[SceneRecord] = []

            for sc, r1, r2, diff in scored_pairs:
                if len(kept_records) >= self.max_pairs_per_target:
                    break

                candidate = (r1, r2)
                too_similar = False
                for prev in kept_pairs:
                    if _ref_pairs_are_similar(
                        candidate, prev, positions,
                        pos_thresh=self.pair_similarity_thresh,
                    ):
                        too_similar = True
                        break
                if too_similar:
                    continue

                tgt_caption = scene_captions.get(resolved[tgt.id], "")
                rec = SceneRecord(
                    scene_name=scene_name, images_dir=images_dir,
                    ref1_name=resolved[r1], ref2_name=resolved[r2],
                    target_name=resolved[tgt.id], prompt=tgt_caption,
                    ref1_c2w=images_by_id[r1].c2w.astype(np.float32),
                    ref2_c2w=images_by_id[r2].c2w.astype(np.float32),
                    tgt_c2w=tgt.c2w.astype(np.float32),
                    K_ref1=K_by_id[r1], K_ref2=K_by_id[r2],
                    K_tgt=K_by_id[tgt.id], score=sc,
                    scene_scale=scene_scale,
                    difficulty=diff,
                )
                kept_pairs.append(candidate)
                kept_records.append(rec)

            if kept_records:
                per_target[tgt.id] = kept_records

        # Round-robin over targets to fill scene budget
        selected: list[SceneRecord] = []
        cursors = {tid: 0 for tid in per_target}
        active = list(per_target.keys())

        while len(selected) < max_triplets and active:
            next_active = []
            for tid in active:
                if len(selected) >= max_triplets:
                    break
                recs = per_target[tid]
                idx = cursors[tid]
                if idx < len(recs):
                    selected.append(recs[idx])
                    cursors[tid] = idx + 1
                    if idx + 1 < len(recs):
                        next_active.append(tid)
            active = next_active

        if selected:
            bc = {d: 0 for d in Difficulty}
            for t in selected:
                bc[t.difficulty] += 1
            # Log raw and normalized translation stats
            raw_dists = []
            for t in selected:
                raw_dists.append(float(np.linalg.norm(t.ref2_c2w[:3, 3] - t.ref1_c2w[:3, 3])))
                raw_dists.append(float(np.linalg.norm(t.tgt_c2w[:3, 3] - t.ref1_c2w[:3, 3])))
            print(f"  {scene_name}: {len(selected)} records "
                  f"(E={bc[Difficulty.EASY]}, M={bc[Difficulty.MEDIUM]}, "
                  f"H={bc[Difficulty.HARD]}) | "
                  f"scale={scene_scale:.4f}, "
                  f"raw_t=[{min(raw_dists):.3f}, {max(raw_dists):.3f}], "
                  f"norm_t=[{2.0*min(raw_dists)/(scene_scale+1e-8):.3f}, "
                  f"{2.0*max(raw_dists)/(scene_scale+1e-8):.3f}]")

        return selected

    # ---- torch Dataset interface -------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        ref1_img, _, _ = load_image_tensor(rec.images_dir, rec.ref1_name,
                                           self.H, self.W)
        ref2_img, _, _ = load_image_tensor(rec.images_dir, rec.ref2_name,
                                           self.H, self.W)
        target_img, _, _ = load_image_tensor(rec.images_dir, rec.target_name,
                                             self.H, self.W)

        # Normalize translations relative to ref1 by scene scale
        ref2_c2w_norm = normalize_c2w_translation(
            rec.ref1_c2w, rec.ref2_c2w, rec.scene_scale)
        tgt_c2w_norm = normalize_c2w_translation(
            rec.ref1_c2w, rec.tgt_c2w, rec.scene_scale)

        plucker_ref1 = compute_plucker_tensor(
            rec.ref1_c2w, rec.ref1_c2w, rec.K_ref1,
            self.H, self.W, self.latent_h, self.latent_w)
        plucker_ref2 = compute_plucker_tensor(
            rec.ref1_c2w, ref2_c2w_norm, rec.K_ref2,
            self.H, self.W, self.latent_h, self.latent_w)
        plucker_tgt = compute_plucker_tensor(
            rec.ref1_c2w, tgt_c2w_norm, rec.K_tgt,
            self.H, self.W, self.latent_h, self.latent_w)

        return {
            "ref1_img": ref1_img,
            "ref2_img": ref2_img,
            "target_img": target_img,
            "plucker_ref1": plucker_ref1,
            "plucker_ref2": plucker_ref2,
            "plucker_tgt": plucker_tgt,
            "prompt": rec.prompt,
            "caption": rec.prompt,
            "scene_name": rec.scene_name,
            "ref1_name": rec.ref1_name,
            "ref2_name": rec.ref2_name,
            "target_name": rec.target_name,
            "difficulty": rec.difficulty.value,
        }


# Backward-compat alias
MegaScenesTriplets = MegaScenesDataset
