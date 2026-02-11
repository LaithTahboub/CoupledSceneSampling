"""
MegaScenes dataset for pose-conditioned SD training.
Precomputes all images and Pluckers at init; __getitem__ is pure indexing.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from css.data.colmap_reader import read_scene, Camera, ImageData
from seva.geometry import get_plucker_coordinates, to_hom_pose


class MegaScenesDataset(Dataset):

    def __init__(
        self,
        scene_dirs: list[str],
        H: int = 512,
        W: int = 512,
        max_pair_distance: float = 2.0,
        max_triplets_per_scene: int = 10000,
        min_dir_similarity: float = 0.3,
        min_ref_spacing: float = 0.3,
    ):
        """
        Args:
            max_pair_distance: Max distance from target to references
            min_dir_similarity: Min dot product of viewing directions (0.3 ≈ 73° angle)
            min_ref_spacing: Min distance between ref1 and ref2 for diversity
        """
        self.H, self.W = H, W
        self.latent_h, self.latent_w = H // 8, W // 8

        raw_triplets = []
        for scene_dir in scene_dirs:
            raw_triplets.extend(
                self._build_triplets(
                    Path(scene_dir), max_pair_distance, max_triplets_per_scene,
                    min_dir_similarity, min_ref_spacing
                )
            )

        self.n = len(raw_triplets)
        if self.n == 0:
            print("MegaScenesDataset: 0 triplets found")
            return

        # Preload unique images
        image_map = {}
        image_list = []
        for images_dir, ref1, _, ref2, _, tgt, _ in raw_triplets:
            for img in (ref1, ref2, tgt):
                key = str(images_dir / img.name)
                if key not in image_map:
                    image_map[key] = len(image_list)
                    image_list.append(self._load_image(images_dir, img))

        self.images = torch.stack(image_list)

        # Precompute Pluckers (deduplicated)
        plucker_cache = {}
        plucker_list = []
        ref1_img_idx, ref2_img_idx, tgt_img_idx = [], [], []
        ref1_plk_idx, ref2_plk_idx, tgt_plk_idx = [], [], []

        for images_dir, ref1, cam1, ref2, cam2, tgt, cam_tgt in tqdm(raw_triplets, desc="Precomputing Pluckers"):
            ref1_img_idx.append(image_map[str(images_dir / ref1.name)])
            ref2_img_idx.append(image_map[str(images_dir / ref2.name)])
            tgt_img_idx.append(image_map[str(images_dir / tgt.name)])

            for img, cam in [(ref1, cam1), (ref2, cam2), (tgt, cam_tgt)]:
                pkey = (tgt.id, img.id, img.camera_id)
                if pkey not in plucker_cache:
                    plucker_cache[pkey] = len(plucker_list)
                    plucker_list.append(self._compute_plucker(tgt.c2w, img.c2w, self._build_K(cam)))

            ref1_plk_idx.append(plucker_cache[(tgt.id, ref1.id, ref1.camera_id)])
            ref2_plk_idx.append(plucker_cache[(tgt.id, ref2.id, ref2.camera_id)])
            tgt_plk_idx.append(plucker_cache[(tgt.id, tgt.id, tgt.camera_id)])

        self.pluckers = torch.stack(plucker_list)
        self.ref1_img_idx = torch.tensor(ref1_img_idx, dtype=torch.long)
        self.ref2_img_idx = torch.tensor(ref2_img_idx, dtype=torch.long)
        self.tgt_img_idx = torch.tensor(tgt_img_idx, dtype=torch.long)
        self.ref1_plk_idx = torch.tensor(ref1_plk_idx, dtype=torch.long)
        self.ref2_plk_idx = torch.tensor(ref2_plk_idx, dtype=torch.long)
        self.tgt_plk_idx = torch.tensor(tgt_plk_idx, dtype=torch.long)

        print(f"Dataset ready: {self.n} triplets, {len(image_list)} images, {len(plucker_list)} unique Pluckers")

    def _build_triplets(self, scene_dir, max_dist, max_triplets, min_dir_sim, min_ref_spacing):
        """Build triplets where refs are closest to target with good FOV overlap."""
        cameras, images = read_scene(scene_dir)
        images_dir = scene_dir / "images"

        valid = [img for img in images.values() if (images_dir / img.name).exists()]
        if len(valid) < 3:
            return []

        triplets = []

        # For each potential target, find best reference pairs
        for target in valid:
            if len(triplets) >= max_triplets:
                break

            target_pos = target.c2w[:3, 3]
            target_dir = self._get_viewing_direction(target.c2w)

            # Score all other images as potential references
            candidates = []
            for ref in valid:
                if ref.id == target.id:
                    continue

                ref_pos = ref.c2w[:3, 3]
                ref_dir = self._get_viewing_direction(ref.c2w)

                distance = np.linalg.norm(ref_pos - target_pos)
                if distance > max_dist:
                    continue

                dir_sim = np.dot(target_dir, ref_dir)
                if dir_sim < min_dir_sim:  # Reject opposing directions
                    continue

                # Combined score: lower is better
                score = distance * (2.0 - dir_sim)
                candidates.append((score, ref, distance, dir_sim))

            if len(candidates) < 2:
                continue

            candidates.sort(key=lambda x: x[0])

            # Generate multiple reference pairs from top candidates
            top_k = min(20, len(candidates))
            for i in range(min(top_k - 1, 5)):  # Up to 5 pairs per target
                if len(triplets) >= max_triplets:
                    break

                ref1_score, ref1, ref1_dist, ref1_sim = candidates[i]

                # Find ref2 that's spatially diverse from ref1
                ref2 = None
                for j in range(i + 1, top_k):
                    score, ref, dist, sim = candidates[j]
                    ref1_to_ref2 = np.linalg.norm(ref.c2w[:3, 3] - ref1.c2w[:3, 3])
                    if ref1_to_ref2 >= min_ref_spacing:
                        ref2 = ref
                        break

                # Fallback: take next best if all are too close
                if ref2 is None and i + 1 < len(candidates):
                    ref2 = candidates[i + 1][1]

                if ref2 is not None:
                    triplets.append((
                        images_dir,
                        ref1, cameras[ref1.camera_id],
                        ref2, cameras[ref2.camera_id],
                        target, cameras[target.camera_id],
                    ))

        return triplets

    def _get_viewing_direction(self, c2w: np.ndarray) -> np.ndarray:
        """Camera forward direction (negative z-axis in camera coords)."""
        return -c2w[:3, 2]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "ref1_img": self.images[self.ref1_img_idx[idx]],
            "ref2_img": self.images[self.ref2_img_idx[idx]],
            "target_img": self.images[self.tgt_img_idx[idx]],
            "plucker_ref1": self.pluckers[self.ref1_plk_idx[idx]],
            "plucker_ref2": self.pluckers[self.ref2_plk_idx[idx]],
            "plucker_target": self.pluckers[self.tgt_plk_idx[idx]],
        }

    def _load_image(self, images_dir: Path, img: ImageData) -> torch.Tensor:
        """Load image to [-1, 1] tensor."""
        pil = Image.open(images_dir / img.name).convert("RGB").resize((self.W, self.H))
        return torch.from_numpy(np.array(pil, dtype=np.float32) / 255.0).permute(2, 0, 1) * 2 - 1

    def _build_K(self, cam: Camera) -> np.ndarray:
        """Scale intrinsics to target resolution."""
        K = cam.K.copy()
        K[0] *= self.W / cam.width
        K[1] *= self.H / cam.height
        return K

    def _compute_plucker(self, c2w_src: np.ndarray, c2w_tgt: np.ndarray, K_tgt: np.ndarray) -> torch.Tensor:
        """Plucker coordinates for tgt view relative to src."""
        c2w_s = to_hom_pose(torch.from_numpy(c2w_src).float().unsqueeze(0))
        c2w_t = to_hom_pose(torch.from_numpy(c2w_tgt).float().unsqueeze(0))
        w2c_s = torch.linalg.inv(c2w_s)
        w2c_t = torch.linalg.inv(c2w_t)
        K_t = torch.from_numpy(K_tgt).float().unsqueeze(0)
        plucker = get_plucker_coordinates(
            extrinsics_src=w2c_s[0], extrinsics=w2c_t,
            intrinsics=K_t, target_size=[self.latent_h, self.latent_w]
        )
        return plucker[0]
