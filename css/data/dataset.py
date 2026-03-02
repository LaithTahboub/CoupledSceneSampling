"""dataset helpers for pose-conditioned sd.

notes:
- intrinsics are normalized to [0, 1] before calling get_plucker_coordinates.
- images are center-cropped then resized so geometry is preserved.
- plucker rays are computed per view and can be anchored to any camera pose.
"""

from itertools import combinations
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


def _frustum_vertices_world(
    c2w: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    near: float,
    far: float,
) -> np.ndarray:
    """Build 8 frustum corner vertices in world coordinates.

    Vertex order:
      0..3 near plane corners (tl, tr, br, bl)
      4..7 far  plane corners (tl, tr, br, bl)
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if abs(fx) < 1e-9 or abs(fy) < 1e-9:
        return np.zeros((0, 3), dtype=np.float64)

    # Use image bounds (not pixel centers) so frustum matches the full raster.
    pix = np.array(
        [
            [0.0, 0.0],
            [float(W), 0.0],
            [float(W), float(H)],
            [0.0, float(H)],
        ],
        dtype=np.float64,
    )

    def backproject(depth: float) -> np.ndarray:
        x = (pix[:, 0] - cx) / fx * depth
        y = (pix[:, 1] - cy) / fy * depth
        z = np.full((4,), depth, dtype=np.float64)
        return np.stack([x, y, z], axis=1)

    near_cam = backproject(float(near))
    far_cam = backproject(float(far))
    cam_pts = np.concatenate([near_cam, far_cam], axis=0)  # (8, 3)

    R = c2w[:3, :3].astype(np.float64)
    t = c2w[:3, 3].astype(np.float64)
    return (R @ cam_pts.T).T + t[None, :]


def _frustum_planes(vertices: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Get outward-facing half-space planes for a convex frustum.

    Each plane is represented as (n, d) with inside inequality:
        n . x + d <= 0
    """
    if vertices.shape[0] != 8:
        return []

    faces = [
        (0, 1, 2, 3),  # near
        (4, 5, 6, 7),  # far
        (0, 3, 7, 4),  # left
        (1, 5, 6, 2),  # right
        (0, 4, 5, 1),  # top
        (3, 2, 6, 7),  # bottom
    ]
    centroid = vertices.mean(axis=0)
    planes: list[tuple[np.ndarray, float]] = []

    for i0, i1, i2, _ in faces:
        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-12:
            continue
        d = -float(np.dot(n, p0))
        # Flip so centroid stays inside (<= 0).
        if float(np.dot(n, centroid) + d) > 0:
            n = -n
            d = -d
        planes.append((n, d))
    return _dedupe_planes(planes, tol=1e-9)


def _normalize_plane(n: np.ndarray, d: float) -> tuple[np.ndarray, float] | None:
    n = np.asarray(n, dtype=np.float64)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        return None
    n = n / n_norm
    d = float(d) / n_norm

    # Canonical sign for deterministic dedupe.
    for comp in n:
        if abs(comp) > 1e-10:
            if comp < 0:
                n = -n
                d = -d
            break
    return n, d


def _dedupe_planes(
    planes: list[tuple[np.ndarray, float]],
    tol: float = 1e-7,
) -> list[tuple[np.ndarray, float]]:
    unique: list[tuple[np.ndarray, float]] = []
    for n_raw, d_raw in planes:
        normalized = _normalize_plane(n_raw, d_raw)
        if normalized is None:
            continue
        n, d = normalized
        exists = False
        for n_u, d_u in unique:
            if np.allclose(n, n_u, atol=tol, rtol=0.0) and abs(d - d_u) <= tol:
                exists = True
                break
        if not exists:
            unique.append((n, d))
    return unique


def _halfspace_intersection_vertices(
    planes: list[tuple[np.ndarray, float]],
    inside_tol: float = 1e-7,
    dedupe_tol: float = 1e-5,
) -> np.ndarray:
    """Enumerate vertices of a convex polyhedron from half-space triplets."""
    if len(planes) < 4:
        return np.zeros((0, 3), dtype=np.float64)

    candidates: list[np.ndarray] = []
    for i, j, k in combinations(range(len(planes)), 3):
        n1, d1 = planes[i]
        n2, d2 = planes[j]
        n3, d3 = planes[k]
        A = np.stack([n1, n2, n3], axis=0)
        if abs(float(np.linalg.det(A))) < 1e-10:
            continue
        b = -np.array([d1, d2, d3], dtype=np.float64)
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        if all(float(np.dot(n, x) + d) <= inside_tol for n, d in planes):
            candidates.append(x)

    if not candidates:
        return np.zeros((0, 3), dtype=np.float64)

    unique: list[np.ndarray] = []
    for x in candidates:
        if not any(float(np.linalg.norm(x - y)) <= dedupe_tol for y in unique):
            unique.append(x)
    return np.asarray(unique, dtype=np.float64)


def _cross2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _convex_hull_2d(points: np.ndarray) -> list[int]:
    n = points.shape[0]
    if n <= 1:
        return list(range(n))

    order = sorted(range(n), key=lambda i: (float(points[i, 0]), float(points[i, 1])))
    lower: list[int] = []
    for idx in order:
        while len(lower) >= 2 and _cross2(points[lower[-2]], points[lower[-1]], points[idx]) <= 1e-12:
            lower.pop()
        lower.append(idx)

    upper: list[int] = []
    for idx in reversed(order):
        while len(upper) >= 2 and _cross2(points[upper[-2]], points[upper[-1]], points[idx]) <= 1e-12:
            upper.pop()
        upper.append(idx)

    hull = lower[:-1] + upper[:-1]
    deduped: list[int] = []
    seen: set[int] = set()
    for idx in hull:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return deduped


def _order_face_vertices(face_vertices: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    if face_vertices.shape[0] < 3:
        return np.zeros((0, 3), dtype=np.float64)

    n = np.asarray(plane_normal, dtype=np.float64)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        return np.zeros((0, 3), dtype=np.float64)
    n = n / n_norm

    centroid = face_vertices.mean(axis=0)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, n))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(n, ref)
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-12:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        u = np.cross(n, ref)
        u_norm = float(np.linalg.norm(u))
        if u_norm < 1e-12:
            return np.zeros((0, 3), dtype=np.float64)
    u = u / u_norm
    v = np.cross(n, u)

    rel = face_vertices - centroid[None, :]
    coords = np.stack([rel @ u, rel @ v], axis=1)
    hull_idx = _convex_hull_2d(coords)
    if len(hull_idx) < 3:
        return np.zeros((0, 3), dtype=np.float64)

    ordered = face_vertices[hull_idx]

    # Ensure CCW when viewed from outside (aligned with outward normal).
    poly_normal = np.zeros((3,), dtype=np.float64)
    for i in range(ordered.shape[0]):
        poly_normal += np.cross(ordered[i], ordered[(i + 1) % ordered.shape[0]])
    if float(np.dot(poly_normal, n)) < 0:
        ordered = ordered[::-1]
    return ordered


def _polyhedron_volume_from_planes(
    planes: list[tuple[np.ndarray, float]],
    vertices: np.ndarray,
    plane_tol: float = 1e-6,
) -> float:
    """Volume of a convex polyhedron from its supporting planes + vertices."""
    if vertices.shape[0] < 4 or not planes:
        return 0.0

    total = 0.0
    for n, d in planes:
        distances = vertices @ n + d
        face_vertices = vertices[np.abs(distances) <= plane_tol]
        if face_vertices.shape[0] < 3:
            continue
        ordered = _order_face_vertices(face_vertices, n)
        if ordered.shape[0] < 3:
            continue
        p0 = ordered[0]
        for i in range(1, ordered.shape[0] - 1):
            p1 = ordered[i]
            p2 = ordered[i + 1]
            total += float(np.dot(p0, np.cross(p1, p2)) / 6.0)
    return float(abs(total))


def build_camera_frustum_geometry(
    c2w: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    d_ref: float,
    near_scale: float = 0.1,
    far_scale: float = 2.5,
) -> dict[str, object]:
    """Build truncated frustum geometry used for true volumetric IoU."""
    d_scene = max(float(d_ref), 1e-3)
    near = max(1e-3, float(near_scale) * d_scene)
    far = max(near + 1e-3, float(far_scale) * d_scene)
    vertices = _frustum_vertices_world(c2w, K, H, W, near, far)
    planes = _frustum_planes(vertices)
    volume = _polyhedron_volume_from_planes(planes, vertices)
    return {
        "near": near,
        "far": far,
        "vertices": vertices,
        "planes": planes,
        "volume": volume,
    }


def compute_frustum_iou_from_geometries(
    geom_i: dict[str, object],
    geom_j: dict[str, object],
) -> float:
    """True IoU between two truncated camera frustum volumes."""
    vol_i = float(geom_i.get("volume", 0.0))
    vol_j = float(geom_j.get("volume", 0.0))
    if vol_i <= 1e-12 or vol_j <= 1e-12:
        return 0.0

    planes_i = geom_i.get("planes", [])
    planes_j = geom_j.get("planes", [])
    if not planes_i or not planes_j:
        return 0.0

    inter_planes = _dedupe_planes(list(planes_i) + list(planes_j))
    inter_vertices = _halfspace_intersection_vertices(inter_planes)
    if inter_vertices.shape[0] < 4:
        return 0.0
    inter_vol = _polyhedron_volume_from_planes(inter_planes, inter_vertices)
    if inter_vol <= 1e-12:
        return 0.0

    union = vol_i + vol_j - inter_vol
    if union <= 1e-12:
        return 0.0
    iou = inter_vol / union
    return float(np.clip(iou, 0.0, 1.0))


def compute_frustum_iou(
    c2w_i: np.ndarray,
    K_i: np.ndarray,
    c2w_j: np.ndarray,
    K_j: np.ndarray,
    H: int,
    W: int,
    d_ref: float,
    near_scale: float = 0.1,
    far_scale: float = 2.5,
) -> float:
    """True IoU between two truncated frustum volumes."""
    geom_i = build_camera_frustum_geometry(c2w_i, K_i, H, W, d_ref, near_scale=near_scale, far_scale=far_scale)
    geom_j = build_camera_frustum_geometry(c2w_j, K_j, H, W, d_ref, near_scale=near_scale, far_scale=far_scale)
    return compute_frustum_iou_from_geometries(geom_i, geom_j)


class MegaScenesDataset(Dataset):
    """Lazy dataset: load images and compute Plucker maps per sample."""

    def __init__(
        self,
        scene_dirs: list[str],
        H: int = 512,
        W: int = 512,
        max_pair_distance: float = 2.0,
        max_triplets_per_scene: int = 8,
        min_pair_iou: float = 0.22,
        min_ref_spacing: float = 0.35,
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

    def _build_triplets(
        self,
        scene_dir: Path,
        scene_key: str,
        max_pair_distance: float,
        max_triplets: int,
        min_pair_iou: float,
        min_ref_spacing: float,
    ):
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
        frustum_by_id: dict[int, dict[str, object]] = {
            img.id: build_camera_frustum_geometry(img.c2w, K_by_id[img.id], self.H, self.W, d_ref)
            for img in valid
        }
        iou_cache: dict[tuple[int, int], float] = {}

        def pair_iou(i_id: int, j_id: int) -> float:
            key = (i_id, j_id) if i_id < j_id else (j_id, i_id)
            if key not in iou_cache:
                iou_cache[key] = compute_frustum_iou_from_geometries(
                    frustum_by_id[key[0]],
                    frustum_by_id[key[1]],
                )
            return iou_cache[key]

        triplets_scored: list[tuple[float, tuple]] = []
        for ref1 in refs_pool:
            ref1_pos = positions[ref1.id]

            # Candidate ref2 cameras (strict only, no fallback).
            ref2_candidates: list[tuple[float, object]] = []
            for ref2 in refs_pool:
                if ref2.id == ref1.id:
                    continue
                dist = float(np.linalg.norm(positions[ref2.id] - ref1_pos))
                if dist < min_ref_spacing:
                    continue
                if max_pair_distance > 0 and dist > max_pair_distance:
                    continue
                iou = pair_iou(ref1.id, ref2.id)
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
                if max_pair_distance > 0 and dist > max_pair_distance:
                    continue
                iou = pair_iou(ref1.id, tgt.id)
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
