"""Camera-frustum geometry and true volumetric IoU utilities."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def compute_reference_depth(positions: dict[int, np.ndarray]) -> float:
    """Median distance from camera positions to their centroid."""
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
    """Build 8 frustum corner vertices in world coordinates."""
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    if abs(fx) < 1e-9 or abs(fy) < 1e-9:
        return np.zeros((0, 3), dtype=np.float64)

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
    cam_pts = np.concatenate([near_cam, far_cam], axis=0)

    R = c2w[:3, :3].astype(np.float64)
    t = c2w[:3, 3].astype(np.float64)
    return (R @ cam_pts.T).T + t[None, :]


def _normalize_plane(n: np.ndarray, d: float) -> tuple[np.ndarray, float] | None:
    n = np.asarray(n, dtype=np.float64)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        return None
    n = n / n_norm
    d = float(d) / n_norm

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


def _frustum_planes(vertices: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Get outward-facing half-space planes for a convex frustum."""
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
        if float(np.linalg.norm(n)) < 1e-12:
            continue
        d = -float(np.dot(n, p0))
        if float(np.dot(n, centroid) + d) > 0:
            n = -n
            d = -d
        planes.append((n, d))
    return _dedupe_planes(planes, tol=1e-9)


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

    if vertices.shape[0] > 0:
        aabb_min = vertices.min(axis=0)
        aabb_max = vertices.max(axis=0)
    else:
        aabb_min = np.zeros((3,), dtype=np.float64)
        aabb_max = np.zeros((3,), dtype=np.float64)

    return {
        "near": near,
        "far": far,
        "vertices": vertices,
        "planes": planes,
        "volume": volume,
        "aabb_min": aabb_min,
        "aabb_max": aabb_max,
    }


def _aabb_disjoint(geom_i: dict[str, object], geom_j: dict[str, object]) -> bool:
    min_i = np.asarray(geom_i["aabb_min"], dtype=np.float64)
    max_i = np.asarray(geom_i["aabb_max"], dtype=np.float64)
    min_j = np.asarray(geom_j["aabb_min"], dtype=np.float64)
    max_j = np.asarray(geom_j["aabb_max"], dtype=np.float64)
    if np.any(max_i < min_j):
        return True
    if np.any(max_j < min_i):
        return True
    return False


def _separated_by_planes(
    planes: list[tuple[np.ndarray, float]],
    other_vertices: np.ndarray,
) -> bool:
    if other_vertices.shape[0] == 0:
        return True
    for n, d in planes:
        if np.all(other_vertices @ n + d > 1e-7):
            return True
    return False


def compute_frustum_iou_from_geometries(
    geom_i: dict[str, object],
    geom_j: dict[str, object],
) -> float:
    """True IoU between two truncated camera frustum volumes."""
    vol_i = float(geom_i.get("volume", 0.0))
    vol_j = float(geom_j.get("volume", 0.0))
    if vol_i <= 1e-12 or vol_j <= 1e-12:
        return 0.0

    if _aabb_disjoint(geom_i, geom_j):
        return 0.0

    planes_i = geom_i.get("planes", [])
    planes_j = geom_j.get("planes", [])
    vertices_i = np.asarray(geom_i.get("vertices", np.zeros((0, 3), dtype=np.float64)))
    vertices_j = np.asarray(geom_j.get("vertices", np.zeros((0, 3), dtype=np.float64)))
    if not planes_i or not planes_j:
        return 0.0

    # Fast SAT-like rejection using existing frustum planes.
    if _separated_by_planes(planes_i, vertices_j) or _separated_by_planes(planes_j, vertices_i):
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
    return float(np.clip(inter_vol / union, 0.0, 1.0))


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
