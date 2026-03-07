"""Triplet dataset with Plucker ray conditioning for PoseSD training.

Extends SingleRefPairDataset to (ref1, ref2, target) triplets with
ref1-anchored Plucker maps. Reuses utilities from css.data.dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from css.data.colmap_reader import read_scene
from css.data.dataset import (
    build_cropped_scaled_intrinsics,
    clean_scene_prompt_name,
    compute_plucker_tensor,
    load_image_tensor,
    read_scene_prompt_name,
)
from css.data.iou import compute_covisibility


@dataclass
class TripletRecord:
    scene_name: str
    images_dir: Path
    ref1_name: str
    ref2_name: str
    target_name: str
    prompt: str
    # Poses and intrinsics for Plucker computation
    ref1_c2w: np.ndarray  # (3, 4) or (4, 4)
    ref2_c2w: np.ndarray
    tgt_c2w: np.ndarray
    K_ref1: np.ndarray  # (3, 3) adjusted for crop+resize
    K_ref2: np.ndarray
    K_tgt: np.ndarray
    score: float


class MegaScenesTriplets(Dataset):
    """Builds (ref1, ref2, target) triplets with co-visibility filtering and Plucker maps."""

    def __init__(
        self,
        scene_dirs: list[str],
        *,
        H: int = 512,
        W: int = 512,
        min_covisibility: float = 0.15,
        max_covisibility: float = 0.55,
        min_ref_covisibility: float = 0.10,
        max_ref_covisibility: float = 0.70,
        min_distance: float = 0.10,
        max_triplets_per_scene: int = 64,
    ):
        self.H, self.W = H, W
        self.latent_h, self.latent_w = H // 8, W // 8
        self.triplets: list[TripletRecord] = []

        for scene_spec in scene_dirs:
            scene_dir = Path(scene_spec)
            triplets = self._build_scene_triplets(
                scene_dir, min_covisibility, max_covisibility,
                min_ref_covisibility, max_ref_covisibility,
                min_distance, max_triplets_per_scene,
            )
            self.triplets.extend(triplets)

        if self.triplets:
            print(f"MegaScenesTriplets: {len(self.triplets)} triplets from {len(scene_dirs)} scenes")
        else:
            print("MegaScenesTriplets: 0 valid triplets found.")

    def _build_scene_triplets(
        self, scene_dir: Path,
        min_covis: float, max_covis: float,
        min_ref_covis: float, max_ref_covis: float,
        min_dist: float, max_triplets: int,
    ) -> list[TripletRecord]:
        cameras, images = read_scene(scene_dir)
        images_dir = scene_dir / "images"
        scene_name = scene_dir.name
        prompt = f"a photo of {clean_scene_prompt_name(read_scene_prompt_name(scene_dir))}"

        # Resolve image names to actual files on disk
        disk_files: set[str] = set()
        for p in images_dir.rglob("*"):
            if p.is_file():
                disk_files.add(p.relative_to(images_dir).as_posix())

        valid = []
        resolved: dict[int, str] = {}
        for img in images.values():
            name = img.name.replace("\\", "/")
            if name in disk_files:
                resolved[img.id] = name
                valid.append(img)

        if len(valid) < 3:
            return []

        positions = {img.id: img.c2w[:3, 3].astype(np.float64) for img in valid}
        images_by_id = {img.id: img for img in valid}

        # Cache pairwise covisibility and distance
        covis_cache: dict[tuple[int, int], float] = {}
        dist_cache: dict[tuple[int, int], float] = {}

        def pair_key(a: int, b: int) -> tuple[int, int]:
            return (a, b) if a < b else (b, a)

        def get_covis(a: int, b: int) -> float:
            k = pair_key(a, b)
            if k not in covis_cache:
                covis_cache[k] = compute_covisibility(images_by_id[k[0]], images_by_id[k[1]])
            return covis_cache[k]

        def get_dist(a: int, b: int) -> float:
            k = pair_key(a, b)
            if k not in dist_cache:
                dist_cache[k] = float(np.linalg.norm(positions[k[0]] - positions[k[1]]))
            return dist_cache[k]

        K_by_id = {
            img.id: build_cropped_scaled_intrinsics(cameras[img.camera_id], self.H, self.W).astype(np.float32)
            for img in valid
        }

        scored: list[TripletRecord] = []

        for tgt in valid:
            # Find refs with good covisibility to target
            ref_candidates: list[tuple[float, object]] = []
            for ref in valid:
                if ref.id == tgt.id:
                    continue
                if get_dist(ref.id, tgt.id) < min_dist:
                    continue
                covis = get_covis(ref.id, tgt.id)
                if covis < min_covis or covis > max_covis:
                    continue
                ref_candidates.append((covis, ref))

            if len(ref_candidates) < 2:
                continue
            ref_candidates.sort(key=lambda x: x[0], reverse=True)

            # Pick best ref1, then find ref2 with bounded ref-ref covisibility
            cov1, ref1 = ref_candidates[0]
            best_ref2 = None
            for cov2, ref2 in ref_candidates[1:]:
                if get_dist(ref1.id, ref2.id) < min_dist:
                    continue
                cov_refs = get_covis(ref1.id, ref2.id)
                if cov_refs < min_ref_covis or cov_refs > max_ref_covis:
                    continue
                if best_ref2 is None or cov2 > best_ref2[0]:
                    best_ref2 = (cov2, ref2, cov_refs)

            if best_ref2 is None:
                continue
            cov2, ref2, cov_refs = best_ref2
            score = float(cov1 + cov2 - 0.25 * cov_refs)

            scored.append(TripletRecord(
                scene_name=scene_name, images_dir=images_dir,
                ref1_name=resolved[ref1.id], ref2_name=resolved[ref2.id],
                target_name=resolved[tgt.id], prompt=prompt,
                ref1_c2w=ref1.c2w.astype(np.float32),
                ref2_c2w=ref2.c2w.astype(np.float32),
                tgt_c2w=tgt.c2w.astype(np.float32),
                K_ref1=K_by_id[ref1.id], K_ref2=K_by_id[ref2.id],
                K_tgt=K_by_id[tgt.id], score=score,
            ))

        scored.sort(key=lambda t: t.score, reverse=True)

        # Diverse selection: prefer unique targets first, then fill
        selected: list[TripletRecord] = []
        used_targets: set[str] = set()
        for t in scored:
            if t.target_name not in used_targets:
                used_targets.add(t.target_name)
                selected.append(t)
                if len(selected) >= max_triplets:
                    break

        if len(selected) < max_triplets:
            selected_set = set(id(t) for t in selected)
            for t in scored:
                if id(t) not in selected_set:
                    selected.append(t)
                    if len(selected) >= max_triplets:
                        break

        if selected:
            covs = [get_covis(
                next(img.id for img in valid if resolved.get(img.id) == t.ref1_name),
                next(img.id for img in valid if resolved.get(img.id) == t.target_name),
            ) for t in selected]
            n_targets = len(set(t.target_name for t in selected))
            print(f"  {scene_name}: {len(selected)} triplets ({n_targets} unique targets), "
                  f"covis=[{min(covs):.3f}, {max(covs):.3f}]")
        return selected

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict:
        rec = self.triplets[idx]

        ref1_img, _, _ = load_image_tensor(rec.images_dir, rec.ref1_name, self.H, self.W)
        ref2_img, _, _ = load_image_tensor(rec.images_dir, rec.ref2_name, self.H, self.W)
        target_img, _, _ = load_image_tensor(rec.images_dir, rec.target_name, self.H, self.W)

        # Ref1-anchored Plucker maps
        plucker_ref1 = compute_plucker_tensor(
            rec.ref1_c2w, rec.ref1_c2w, rec.K_ref1,
            self.H, self.W, self.latent_h, self.latent_w,
        )
        plucker_ref2 = compute_plucker_tensor(
            rec.ref1_c2w, rec.ref2_c2w, rec.K_ref2,
            self.H, self.W, self.latent_h, self.latent_w,
        )
        plucker_tgt = compute_plucker_tensor(
            rec.ref1_c2w, rec.tgt_c2w, rec.K_tgt,
            self.H, self.W, self.latent_h, self.latent_w,
        )

        return {
            "ref1_img": ref1_img,
            "ref2_img": ref2_img,
            "target_img": target_img,
            "plucker_ref1": plucker_ref1,
            "plucker_ref2": plucker_ref2,
            "plucker_tgt": plucker_tgt,
            "prompt": rec.prompt,
            "scene_name": rec.scene_name,
            "ref1_name": rec.ref1_name,
            "ref2_name": rec.ref2_name,
            "target_name": rec.target_name,
        }
