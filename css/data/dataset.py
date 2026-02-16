"""MegaScenes dataset for pose-conditioned SD training."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from css.data.colmap_reader import Camera, read_scene
from seva.geometry import get_plucker_coordinates, to_hom_pose


def load_image_name_set(path: str | None) -> set[str] | None:
    if path is None:
        return None
    names: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            names.add(name)
    return names


def scene_image_key(scene_name: str, image_name: str) -> str:
    return f"{scene_name}/{image_name}"


def scene_candidate_keys(scene_name: str, image_name: str) -> tuple[str, str]:
    return image_name, scene_image_key(scene_name, image_name)


def name_allowed(name_filter: set[str] | None, scene_name: str, image_name: str) -> bool:
    if name_filter is None:
        return True
    return any(k in name_filter for k in scene_candidate_keys(scene_name, image_name))


def load_image_tensor(images_dir: Path, image_name: str, H: int, W: int) -> torch.Tensor:
    pil = Image.open(images_dir / image_name).convert("RGB").resize((W, H))
    return torch.from_numpy(np.array(pil, dtype=np.float32) / 255.0).permute(2, 0, 1) * 2 - 1


def build_scaled_intrinsics(cam: Camera, H: int, W: int) -> np.ndarray:
    K = cam.K.copy()
    K[0] *= W / cam.width
    K[1] *= H / cam.height
    return K


def compute_plucker_tensor(
    c2w_src: np.ndarray,
    c2w_tgt: np.ndarray,
    K_tgt: np.ndarray,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    c2w_s = to_hom_pose(torch.from_numpy(c2w_src).float().unsqueeze(0))
    c2w_t = to_hom_pose(torch.from_numpy(c2w_tgt).float().unsqueeze(0))
    w2c_s = torch.linalg.inv(c2w_s)
    w2c_t = torch.linalg.inv(c2w_t)
    K_t = torch.from_numpy(K_tgt).float().unsqueeze(0)
    plucker = get_plucker_coordinates(
        extrinsics_src=w2c_s[0],
        extrinsics=w2c_t,
        intrinsics=K_t,
        target_size=[latent_h, latent_w],
    )
    return plucker[0]


class MegaScenesDataset(Dataset):
    """Lazy dataset: load images and compute Plucker maps per sample."""

    def __init__(
        self,
        scene_dirs: list[str],
        H: int = 512,
        W: int = 512,
        max_pair_distance: float = 2.0,
        max_triplets_per_scene: int = 10000,
        min_dir_similarity: float = 0.3,
        min_ref_spacing: float = 0.3,
        exclude_image_names: set[str] | None = None,
        target_include_image_names: set[str] | None = None,
        reference_include_image_names: set[str] | None = None,
        prompt_template: str | None = None,
    ):
        self.H, self.W = H, W
        self.latent_h, self.latent_w = H // 8, W // 8
        self.exclude_image_names = exclude_image_names
        self.target_include_image_names = target_include_image_names
        self.reference_include_image_names = reference_include_image_names
        self.prompt_template = prompt_template

        self.triplets = []
        for scene_dir in scene_dirs:
            self.triplets.extend(
                self._build_triplets(
                    Path(scene_dir),
                    max_pair_distance,
                    max_triplets_per_scene,
                    min_dir_similarity,
                    min_ref_spacing,
                )
            )

        self.n = len(self.triplets)
        if self.n == 0:
            print("MegaScenesDataset: 0 triplets found")
            return

        unique_images = set()
        scene_names = set()
        for scene_name, images_dir, ref1_name, ref2_name, tgt_name, *_ in self.triplets:
            scene_names.add(scene_name)
            unique_images.add(str(images_dir / ref1_name))
            unique_images.add(str(images_dir / ref2_name))
            unique_images.add(str(images_dir / tgt_name))
        print(f"Dataset ready: {self.n} triplets, {len(unique_images)} unique images, {len(scene_names)} scenes")

    def _build_triplets(self, scene_dir, max_dist, max_triplets, min_dir_sim, min_ref_spacing):
        cameras, images = read_scene(scene_dir)
        images_dir = scene_dir / "images"

        valid = [img for img in images.values() if (images_dir / img.name).exists()]
        valid.sort(key=lambda x: x.id)
        valid = [img for img in valid if not name_allowed(self.exclude_image_names, scene_dir.name, img.name)]

        targets = [img for img in valid if name_allowed(self.target_include_image_names, scene_dir.name, img.name)]
        refs_pool = [img for img in valid if name_allowed(self.reference_include_image_names, scene_dir.name, img.name)]

        print(
            f"{scene_dir.name}: usable images={len(valid)}, "
            f"target_pool={len(targets)}, ref_pool={len(refs_pool)}"
        )

        if len(targets) == 0 or len(refs_pool) < 2:
            return []

        triplets = []

        for target in targets:
            if len(triplets) >= max_triplets:
                break

            target_pos = target.c2w[:3, 3]
            target_dir = self._get_viewing_direction(target.c2w)

            candidates = []
            for ref in refs_pool:
                if ref.id == target.id:
                    continue

                ref_pos = ref.c2w[:3, 3]
                ref_dir = self._get_viewing_direction(ref.c2w)

                distance = np.linalg.norm(ref_pos - target_pos)
                if distance > max_dist:
                    continue

                dir_sim = np.dot(target_dir, ref_dir)
                if dir_sim < min_dir_sim:
                    continue

                score = distance * (2.0 - dir_sim)
                candidates.append((score, ref))

            if len(candidates) < 2:
                continue

            candidates.sort(key=lambda x: x[0])
            top_k = min(20, len(candidates))

            for i in range(min(top_k - 1, 5)):
                if len(triplets) >= max_triplets:
                    break

                ref1 = candidates[i][1]
                ref2 = None

                for j in range(i + 1, top_k):
                    ref = candidates[j][1]
                    ref1_to_ref2 = np.linalg.norm(ref.c2w[:3, 3] - ref1.c2w[:3, 3])
                    if ref1_to_ref2 >= min_ref_spacing:
                        ref2 = ref
                        break

                if ref2 is None and i + 1 < len(candidates):
                    ref2 = candidates[i + 1][1]

                if ref2 is None:
                    continue

                cam1 = cameras[ref1.camera_id]
                cam2 = cameras[ref2.camera_id]
                cam_t = cameras[target.camera_id]

                triplets.append((
                    scene_dir.name,
                    images_dir,
                    ref1.name,
                    ref2.name,
                    target.name,
                    ref1.c2w.astype(np.float32),
                    ref2.c2w.astype(np.float32),
                    target.c2w.astype(np.float32),
                    self._build_K(cam1).astype(np.float32),
                    self._build_K(cam2).astype(np.float32),
                    self._build_K(cam_t).astype(np.float32),
                ))

        return triplets

    @staticmethod
    def _get_viewing_direction(c2w: np.ndarray) -> np.ndarray:
        return -c2w[:3, 2]

    def _scene_prompt(self, scene_name: str) -> str:
        if not self.prompt_template:
            return ""
        scene_text = scene_name.replace("_", " ")
        if "{scene}" in self.prompt_template:
            return self.prompt_template.format(scene=scene_text)
        return self.prompt_template

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        (
            scene_name,
            images_dir,
            ref1_name,
            ref2_name,
            tgt_name,
            ref1_c2w,
            ref2_c2w,
            tgt_c2w,
            K_ref1,
            K_ref2,
            K_tgt,
        ) = self.triplets[idx]

        ref1_img = load_image_tensor(images_dir, ref1_name, self.H, self.W)
        ref2_img = load_image_tensor(images_dir, ref2_name, self.H, self.W)
        target_img = load_image_tensor(images_dir, tgt_name, self.H, self.W)

        plucker_ref1 = self._compute_plucker(ref1_c2w, ref1_c2w, K_ref1)
        plucker_ref2 = self._compute_plucker(ref1_c2w, ref2_c2w, K_ref2)
        plucker_target = self._compute_plucker(ref1_c2w, tgt_c2w, K_tgt)

        out = {
            "ref1_img": ref1_img,
            "ref2_img": ref2_img,
            "target_img": target_img,
            "plucker_ref1": plucker_ref1,
            "plucker_ref2": plucker_ref2,
            "plucker_target": plucker_target,
            "scene_name": scene_name,
        }
        if self.prompt_template:
            out["prompt"] = self._scene_prompt(scene_name)
        return out

    def _build_K(self, cam: Camera) -> np.ndarray:
        return build_scaled_intrinsics(cam, self.H, self.W)

    def _compute_plucker(self, c2w_src: np.ndarray, c2w_tgt: np.ndarray, K_tgt: np.ndarray) -> torch.Tensor:
        return compute_plucker_tensor(c2w_src, c2w_tgt, K_tgt, self.latent_h, self.latent_w)
