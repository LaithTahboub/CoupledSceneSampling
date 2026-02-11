"""Minimal COLMAP binary reader for cameras.bin and images.bin."""

import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass

_MODEL_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12, 6: 5}


@dataclass
class Camera:
    id: int
    model_id: int
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsic matrix."""
        if self.model_id in (0, 2, 3):  # SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL
            f, cx, cy = self.params[0], self.params[1], self.params[2]
            return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        else:
            fx, fy, cx, cy = self.params[0], self.params[1], self.params[2], self.params[3]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


@dataclass
class ImageData:
    id: int
    camera_id: int
    name: str
    c2w: np.ndarray  # 4x4


def _quat_to_rotation(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])


def read_cameras_bin(path: str | Path) -> dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            n_params = _MODEL_PARAMS.get(model_id, 4)
            params = np.array(struct.unpack(f"<{n_params}d", f.read(8 * n_params)))
            cameras[cam_id] = Camera(cam_id, model_id, width, height, params)
    return cameras


def read_images_bin(path: str | Path) -> dict[int, ImageData]:
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]

            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8")

            # Skip 2D points
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)

            R_w2c = _quat_to_rotation(qw, qx, qy, qz)
            t_w2c = np.array([tx, ty, tz])
            c2w = np.eye(4)
            c2w[:3, :3] = R_w2c.T
            c2w[:3, 3] = -R_w2c.T @ t_w2c

            images[image_id] = ImageData(image_id, camera_id, name, c2w)

    return images


def read_scene(scene_dir: str | Path) -> tuple[dict[int, Camera], dict[int, ImageData]]:
    """Read COLMAP cameras.bin and images.bin from scene_dir/sparse/."""
    scene_dir = Path(scene_dir)
    sparse_dir = scene_dir / "sparse"
    return read_cameras_bin(sparse_dir / "cameras.bin"), read_images_bin(sparse_dir / "images.bin")
