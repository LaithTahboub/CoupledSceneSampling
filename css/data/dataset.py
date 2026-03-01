"""dataset helpers for pose-conditioned sd.

notes:
- intrinsics are normalized to [0, 1] before calling get_plucker_coordinates.
- images are center-cropped then resized so geometry is preserved.
- plucker rays are computed per view and can be anchored to any camera pose.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from css.data.colmap_reader import Camera, read_scene
from seva.geometry import get_plucker_coordinates, to_hom_pose

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageLoadError(OSError):
    def __init__(self, path: Path, cause: Exception):
        super().__init__(f"{path}: {cause}")
        self.path = str(path)


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


def name_allowed(
    name_filter: set[str] | None,
    scene_name: str,
    image_name: str,
) -> bool:
    if name_filter is None:
        return True
    return any(k in name_filter for k in scene_candidate_keys(scene_name, image_name))


def name_excluded(
    exclude_filter: set[str] | None,
    scene_name: str,
    image_name: str,
) -> bool:
    if exclude_filter is None:
        return False
    return any(k in exclude_filter for k in scene_candidate_keys(scene_name, image_name))


def clean_scene_prompt_name(scene_name: str) -> str:
    cleaned = (
        scene_name.replace("_", " ")
        .replace('"', "")
        .replace("'", "")
        .replace("/", " ")
    )
    return " ".join(cleaned.split())


def read_scene_prompt_name(scene_dir: Path) -> str:
    raw_name_path = scene_dir / "scene_name_raw.txt"
    if raw_name_path.exists():
        text = raw_name_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return scene_dir.name


def _center_crop_and_resize(pil_img: Image.Image, H: int, W: int) -> tuple[Image.Image, float, float]:
    """Center-crop to target aspect ratio, then resize.

    Returns:
        (cropped_resized_image, crop_offset_x, crop_offset_y)
        where offsets are in *original* pixel coordinates.
    """
    src_w, src_h = pil_img.size
    target_aspect = W / H
    src_aspect = src_w / src_h

    if src_aspect > target_aspect:
        # Source is wider: crop horizontally
        new_w = int(src_h * target_aspect)
        offset_x = (src_w - new_w) // 2
        offset_y = 0
        pil_img = pil_img.crop((offset_x, 0, offset_x + new_w, src_h))
    elif src_aspect < target_aspect:
        # Source is taller: crop vertically
        new_h = int(src_w / target_aspect)
        offset_x = 0
        offset_y = (src_h - new_h) // 2
        pil_img = pil_img.crop((0, offset_y, src_w, offset_y + new_h))
    else:
        offset_x, offset_y = 0, 0

    pil_img = pil_img.resize((W, H), Image.LANCZOS)
    return pil_img, float(offset_x), float(offset_y)


def load_image_tensor(images_dir: Path, image_name: str, H: int, W: int) -> tuple[torch.Tensor, int, int]:
    """Load, center-crop, resize to (H, W).  Returns (tensor, orig_w, orig_h)."""
    path = images_dir / image_name
    try:
        with Image.open(path) as pil:
            rgb = pil.convert("RGB")
            orig_w, orig_h = rgb.size
            cropped, _, _ = _center_crop_and_resize(rgb, H, W)
            arr = np.array(cropped, dtype=np.float32) / 255.0
    except Exception as e:
        raise ImageLoadError(path, e) from e
    tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1
    return tensor, orig_w, orig_h


def _index_scene_images(images_dir: Path) -> tuple[set[str], dict[str, str]]:
    rel_paths: list[str] = []
    for p in images_dir.rglob("*"):
        if p.is_file():
            rel_paths.append(p.relative_to(images_dir).as_posix())

    rel_set = set(rel_paths)
    by_basename: dict[str, list[str]] = {}
    for rel in rel_paths:
        by_basename.setdefault(Path(rel).name, []).append(rel)
    unique_basename = {k: v[0] for k, v in by_basename.items() if len(v) == 1}
    return rel_set, unique_basename


def _resolve_image_name(image_name: str, rel_set: set[str], unique_basename: dict[str, str]) -> str | None:
    norm = image_name.replace("\\", "/")
    if norm in rel_set:
        return norm
    return unique_basename.get(Path(norm).name)


def build_cropped_scaled_intrinsics(cam: Camera, H: int, W: int) -> np.ndarray:
    """Build intrinsics for center-crop + resize from (cam.width, cam.height) to (W, H).

    Steps:
    1. Start with original K.
    2. Adjust principal point for center-crop offset.
    3. Scale for resize from cropped size to (W, H).
    """
    K = cam.K.copy()
    src_w, src_h = cam.width, cam.height
    target_aspect = W / H
    src_aspect = src_w / src_h

    if src_aspect > target_aspect:
        # Horizontal crop
        new_w = int(src_h * target_aspect)
        offset_x = (src_w - new_w) / 2.0
        K[0, 2] -= offset_x  # shift cx
        # Scale from (new_w, src_h) -> (W, H)
        K[0] *= W / new_w
        K[1] *= H / src_h
    elif src_aspect < target_aspect:
        # Vertical crop
        new_h = int(src_w / target_aspect)
        offset_y = (src_h - new_h) / 2.0
        K[1, 2] -= offset_y  # shift cy
        # Scale from (src_w, new_h) -> (W, H)
        K[0] *= W / src_w
        K[1] *= H / new_h
    else:
        # Same aspect: just resize
        K[0] *= W / src_w
        K[1] *= H / src_h

    return K


def normalize_intrinsics(K: np.ndarray, H: int, W: int) -> np.ndarray:
    """Normalize intrinsics to resolution-independent [0,1] coordinates.

    Divides fx, cx by W and fy, cy by H so that principal point is in [0,1].
    """
    K_norm = K.copy()
    K_norm[0, :] /= W  # fx, skew, cx
    K_norm[1, :] /= H  # fy, 0,    cy
    return K_norm


def compute_plucker_tensor(
    c2w_anchor: np.ndarray,
    c2w_ray_cam: np.ndarray,
    K_ray_cam: np.ndarray,
    H: int,
    W: int,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    """Compute a plucker ray map at ray_cam's pixel grid, relative to anchor.

    Args:
        c2w_anchor: 4x4 camera-to-world of the anchor camera.
        c2w_ray_cam: 4x4 camera-to-world of the camera whose pixel grid defines rays.
        K_ray_cam: 3x3 intrinsics of ray_cam, at full image resolution (H, W).
        H, W: full image resolution.
        latent_h, latent_w: spatial size of the output Plucker map.

    Returns:
        Plucker map of shape (6, latent_h, latent_w).
    """
    # Convert c2w -> w2c
    c2w_a = to_hom_pose(torch.from_numpy(c2w_anchor).float().unsqueeze(0))
    c2w_r = to_hom_pose(torch.from_numpy(c2w_ray_cam).float().unsqueeze(0))
    w2c_anchor = torch.linalg.inv(c2w_a)
    w2c_ray = torch.linalg.inv(c2w_r)

    # Normalize intrinsics to [0,1] *before* calling get_plucker_coordinates
    # so we don't depend on the auto-detection heuristic inside SEVA.
    K_norm = normalize_intrinsics(K_ray_cam, H, W)
    K_t = torch.from_numpy(K_norm).float().unsqueeze(0)

    # extrinsics_src = anchor's w2c  (the function inverts it to get c2w_anchor)
    # extrinsics = ray_cam's w2c     (the pixel grid camera, gets relative pose)
    # intrinsics = ray_cam's K       (defines the pixel grid)
    plucker = get_plucker_coordinates(
        extrinsics_src=w2c_anchor[0],
        extrinsics=w2c_ray,
        intrinsics=K_t,
        target_size=[latent_h, latent_w],
    )
    return plucker[0]


def compute_reference_depth(positions: dict[int, np.ndarray]) -> float:
    """Median distance from camera positions to their centroid.

    Provides a scene-adaptive reference depth for frustum IoU computation.
    """
    pos_list = list(positions.values())
    if len(pos_list) < 2:
        return 1.0
    centroid = np.mean(pos_list, axis=0)
    dists = [float(np.linalg.norm(p - centroid)) for p in pos_list]
    return float(np.median(dists))


def compute_frustum_iou(
    c2w_i: np.ndarray,
    K_i: np.ndarray,
    c2w_j: np.ndarray,
    K_j: np.ndarray,
    H: int,
    W: int,
    d_ref: float,
    M: int = 16,
) -> float:
    """Approximate frustum IoU between two cameras at reference depth d_ref.

    Samples an MxM pixel grid in each camera, backprojects to 3D at d_ref,
    projects into the other camera, and counts in-bounds hits.  Combines
    the two directional coverages into an IoU estimate:
        IoU = (r_ij * r_ji) / (r_ij + r_ji - r_ij * r_ji)
    where r_ij = fraction of camera-i samples visible in camera j.
    """
    n = M * M
    u = np.linspace(0.5, W - 0.5, M)
    v = np.linspace(0.5, H - 0.5, M)
    uu, vv = np.meshgrid(u, v)
    pixels = np.stack([uu.ravel(), vv.ravel(), np.ones(n)], axis=0)  # (3, n)

    # -- forward: sample camera i, project into camera j ----------------
    K_i_inv = np.linalg.inv(K_i)
    rays_cam = K_i_inv @ pixels                       # (3, n) camera-i frame
    pts_cam = rays_cam * (d_ref / rays_cam[2:3, :])   # depth along optical axis
    pts_world = c2w_i[:3, :3] @ pts_cam + c2w_i[:3, 3:4]

    R_j_T = c2w_j[:3, :3].T
    t_j = c2w_j[:3, 3]
    pts_cam_j = R_j_T @ (pts_world - t_j[:, None])
    valid = pts_cam_j[2, :] > 0
    z_safe = np.where(valid, pts_cam_j[2, :], 1.0)
    proj = K_j @ pts_cam_j
    u_j = proj[0, :] / z_safe
    v_j = proj[1, :] / z_safe
    hits_ij = valid & (u_j >= 0) & (u_j < W) & (v_j >= 0) & (v_j < H)
    r_ij = hits_ij.sum() / n

    # -- reverse: sample camera j, project into camera i ----------------
    K_j_inv = np.linalg.inv(K_j)
    rays_cam_j = K_j_inv @ pixels
    pts_cam_j2 = rays_cam_j * (d_ref / rays_cam_j[2:3, :])
    pts_world2 = c2w_j[:3, :3] @ pts_cam_j2 + c2w_j[:3, 3:4]

    R_i_T = c2w_i[:3, :3].T
    t_i = c2w_i[:3, 3]
    pts_cam_i = R_i_T @ (pts_world2 - t_i[:, None])
    valid2 = pts_cam_i[2, :] > 0
    z_safe2 = np.where(valid2, pts_cam_i[2, :], 1.0)
    proj2 = K_i @ pts_cam_i
    u_i = proj2[0, :] / z_safe2
    v_i = proj2[1, :] / z_safe2
    hits_ji = valid2 & (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    r_ji = hits_ji.sum() / n

    # -- combine into IoU -----------------------------------------------
    denom = r_ij + r_ji - r_ij * r_ji
    if denom < 1e-12:
        return 0.0
    return float((r_ij * r_ji) / denom)


class MegaScenesDataset(Dataset):
    """Lazy dataset: load images and compute Plucker maps per sample."""

    def __init__(
        self,
        scene_dirs: list[str],
        H: int = 512,
        W: int = 512,
        max_pair_distance: float = 2.0,
        max_triplets_per_scene: int = 8,
        min_pair_iou: float = 0.15,
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
        self.scene_prompt_names: dict[str, str] = {}
        self._bad_image_paths: set[str] = set()
        self._bad_triplet_indices: set[int] = set()
        self._max_io_retries = 64
        self._io_error_logs = 0
        self._max_io_error_logs = 20

        if max_triplets_per_scene <= 0:
            raise ValueError("max_triplets_per_scene must be >= 1")

        self.triplets = []
        for scene_spec in scene_dirs:
            scene_dir = Path(scene_spec)
            scene_key = scene_dir.name
            self.scene_prompt_names[scene_key] = clean_scene_prompt_name(read_scene_prompt_name(scene_dir))
            self.triplets.extend(
                self._build_triplets(
                    scene_dir,
                    scene_key,
                    max_pair_distance,
                    max_triplets_per_scene,
                    min_pair_iou,
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

    def _build_triplets(self, scene_dir: Path, scene_key: str, max_dist, max_triplets, min_pair_iou, min_ref_spacing):
        cameras, images = read_scene(scene_dir)
        images_dir = scene_dir / "images"
        rel_set, unique_basename = _index_scene_images(images_dir)

        resolved_name_by_id: dict[int, str] = {}
        valid = []
        for img in sorted(images.values(), key=lambda x: x.id):
            resolved = _resolve_image_name(img.name, rel_set, unique_basename)
            if resolved is None:
                continue
            if name_excluded(self.exclude_image_names, scene_dir.name, img.name):
                continue
            resolved_name_by_id[img.id] = resolved
            valid.append(img)

        targets = [img for img in valid if name_allowed(self.target_include_image_names, scene_dir.name, img.name)]
        refs_pool = [img for img in valid if name_allowed(self.reference_include_image_names, scene_dir.name, img.name)]

        print(
            f"{scene_key}: usable images={len(valid)}, "
            f"target_pool={len(targets)}, ref_pool={len(refs_pool)}"
        )

        if len(targets) == 0 or len(refs_pool) < 2:
            return []

        positions = {img.id: img.c2w[:3, 3] for img in valid}
        K_by_id = {img.id: self._build_K(cameras[img.camera_id]) for img in valid}
        target_ids = {img.id for img in targets}
        d_ref = compute_reference_depth(positions)

        triplets_scored: list[tuple[float, tuple]] = []
        for ref1 in refs_pool:
            ref1_pos = positions[ref1.id]
            ref1_K = K_by_id[ref1.id]

            # Candidate ref2 cameras (strict only, no fallback).
            ref2_candidates: list[tuple[float, object]] = []
            for ref2 in refs_pool:
                if ref2.id == ref1.id:
                    continue
                dist = float(np.linalg.norm(positions[ref2.id] - ref1_pos))
                if dist < min_ref_spacing:
                    continue
                if max_dist > 0 and dist > max_dist:
                    continue
                iou = compute_frustum_iou(
                    ref1.c2w, ref1_K,
                    ref2.c2w, K_by_id[ref2.id],
                    self.H, self.W, d_ref,
                )
                if iou < min_pair_iou:
                    continue
                ref2_candidates.append((iou, ref2))

            # Candidate targets (strict only, no fallback).
            target_candidates: list[tuple[float, object]] = []
            for tgt in valid:
                if tgt.id == ref1.id:
                    continue
                if tgt.id not in target_ids:
                    continue
                dist = float(np.linalg.norm(positions[tgt.id] - ref1_pos))
                if max_dist > 0 and dist > max_dist:
                    continue
                iou = compute_frustum_iou(
                    ref1.c2w, ref1_K,
                    tgt.c2w, K_by_id[tgt.id],
                    self.H, self.W, d_ref,
                )
                if iou < min_pair_iou:
                    continue
                target_candidates.append((iou, tgt))

            if len(ref2_candidates) == 0 or len(target_candidates) == 0:
                continue

            # Sort by IoU descending (higher = better overlap).
            ref2_candidates.sort(key=lambda x: x[0], reverse=True)
            target_candidates.sort(key=lambda x: x[0], reverse=True)

            best = None
            for ref2_iou, ref2 in ref2_candidates[:20]:
                for tgt_iou, tgt in target_candidates[:20]:
                    if tgt.id == ref2.id:
                        continue
                    combo = ref2_iou + tgt_iou
                    if best is None or combo > best[0]:
                        best = (combo, ref2, tgt)
                    break
            if best is None:
                continue

            _, ref2, target = best
            cam1 = cameras[ref1.camera_id]
            cam2 = cameras[ref2.camera_id]
            cam_t = cameras[target.camera_id]
            triplet = (
                scene_key,
                images_dir,
                resolved_name_by_id[ref1.id],
                resolved_name_by_id[ref2.id],
                resolved_name_by_id[target.id],
                ref1.c2w.astype(np.float32),
                ref2.c2w.astype(np.float32),
                target.c2w.astype(np.float32),
                self._build_K(cam1).astype(np.float32),
                self._build_K(cam2).astype(np.float32),
                self._build_K(cam_t).astype(np.float32),
            )
            triplets_scored.append((best[0], triplet))

        # Sort by score descending (higher IoU = better).
        triplets_scored.sort(key=lambda x: x[0], reverse=True)

        # Prefer target diversity first, then fill remaining slots by score.
        selected_indices: list[int] = []
        used_targets: set[str] = set()
        for i, (_, triplet) in enumerate(triplets_scored):
            target_name = str(triplet[4])
            if target_name in used_targets:
                continue
            used_targets.add(target_name)
            selected_indices.append(i)
            if len(selected_indices) >= max_triplets:
                break

        if len(selected_indices) < max_triplets:
            selected_set = set(selected_indices)
            for i in range(len(triplets_scored)):
                if i in selected_set:
                    continue
                selected_indices.append(i)
                if len(selected_indices) >= max_triplets:
                    break

        return [triplets_scored[i][1] for i in selected_indices]

    def _scene_prompt(self, scene_name: str) -> str:
        if not self.prompt_template:
            return ""
        scene_text = self.scene_prompt_names.get(scene_name, clean_scene_prompt_name(scene_name))
        if "{scene}" in self.prompt_template:
            return self.prompt_template.format(scene=scene_text)
        return self.prompt_template

    def __len__(self):
        return self.n

    def _make_item_from_triplet(self, idx: int):
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

        ref1_path = str(images_dir / ref1_name)
        ref2_path = str(images_dir / ref2_name)
        tgt_path = str(images_dir / tgt_name)
        bad_path = None
        if ref1_path in self._bad_image_paths:
            bad_path = ref1_path
        elif ref2_path in self._bad_image_paths:
            bad_path = ref2_path
        elif tgt_path in self._bad_image_paths:
            bad_path = tgt_path
        if bad_path is not None:
            self._bad_triplet_indices.add(idx)
            raise ImageLoadError(Path(bad_path), RuntimeError("Triplet references previously failed image"))

        ref1_img, _, _ = load_image_tensor(images_dir, ref1_name, self.H, self.W)
        ref2_img, _, _ = load_image_tensor(images_dir, ref2_name, self.H, self.W)
        target_img, _, _ = load_image_tensor(images_dir, tgt_name, self.H, self.W)

        # ref1-anchored pluckers for cat3d-style conditioning.
        plucker_ref1 = self._compute_plucker(ref1_c2w, ref1_c2w, K_ref1)
        plucker_ref2 = self._compute_plucker(ref1_c2w, ref2_c2w, K_ref2)
        plucker_tgt = self._compute_plucker(ref1_c2w, tgt_c2w, K_tgt)

        out = {
            "ref1_img": ref1_img,
            "ref2_img": ref2_img,
            "target_img": target_img,
            "plucker_ref1": plucker_ref1,
            "plucker_ref2": plucker_ref2,
            "plucker_tgt": plucker_tgt,
            "scene_name": scene_name,
        }
        if self.prompt_template:
            out["prompt"] = self._scene_prompt(scene_name)
        return out

    def __getitem__(self, idx):
        if self.n == 0:
            raise IndexError("Empty dataset")

        last_error = None
        for attempt in range(self._max_io_retries):
            candidate_idx = (idx + attempt) % self.n
            if candidate_idx in self._bad_triplet_indices:
                continue
            try:
                return self._make_item_from_triplet(candidate_idx)
            except ImageLoadError as e:
                last_error = e
                self._bad_image_paths.add(e.path)
                self._bad_triplet_indices.add(candidate_idx)
                if self._io_error_logs < self._max_io_error_logs:
                    print(f"[dataset] skipping unreadable image: {e.path}")
                    self._io_error_logs += 1
            except (OSError, FileNotFoundError) as e:
                last_error = e
                self._bad_triplet_indices.add(candidate_idx)
                if self._io_error_logs < self._max_io_error_logs:
                    print(f"[dataset] skipping unreadable triplet idx={candidate_idx}: {e}")
                    self._io_error_logs += 1

        raise RuntimeError(
            f"Failed to load sample after {self._max_io_retries} retries; "
            f"bad_triplets={len(self._bad_triplet_indices)} bad_images={len(self._bad_image_paths)}"
        ) from last_error

    def _build_K(self, cam: Camera) -> np.ndarray:
        """Build intrinsics adjusted for center-crop + resize to (self.W, self.H)."""
        return build_cropped_scaled_intrinsics(cam, self.H, self.W)

    def _compute_plucker(self, c2w_anchor: np.ndarray, c2w_ray_cam: np.ndarray, K_ray_cam: np.ndarray) -> torch.Tensor:
        """Plucker map at ray_cam's pixel grid, anchored to c2w_anchor."""
        return compute_plucker_tensor(
            c2w_anchor, c2w_ray_cam, K_ray_cam,
            self.H, self.W,
            self.latent_h, self.latent_w,
        )
