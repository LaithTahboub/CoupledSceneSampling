"""Coupled SEVA + PoseSD inference on a COLMAP scene.

Generates a SEVA video interpolating between ordered input images, then
replaces frames near each input keyframe with PoseSD-canonicalized versions
using the two closest input images as references.

Usage:
    python -m css.inference.infer_coupled \
        --scene /path/to/MegaScenes/scene \
        --inputs img1.jpg img2.jpg img3.jpg \
        --checkpoint /path/to/pose_sd_unet.pt \
        --output coupled_video.mp4
"""

import argparse
import copy
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from css.data.colmap_reader import read_scene
from css.data.dataset import (
    build_cropped_scaled_intrinsics,
    compute_plucker_tensor,
    load_image_tensor,
)
from css.models.EMA import load_pose_sd_checkpoint
from css.models.pose_sd import PoseSD

import sys

_SEVA_ROOT = str(Path(__file__).resolve().parent.parent.parent / "stable-virtual-camera")
if _SEVA_ROOT not in sys.path:
    sys.path.insert(0, _SEVA_ROOT)

from seva.eval import infer_prior_stats, run_one_scene, transform_img_and_K  # noqa: E402
from seva.geometry import normalize_scene  # noqa: E402
from seva.model import SGMWrapper  # noqa: E402
from seva.modules.autoencoder import AutoEncoder  # noqa: E402
from seva.modules.conditioner import CLIPConditioner  # noqa: E402
from seva.modules.preprocessor import Dust3rPipeline  # noqa: E402
from seva.sampling import DiscreteDenoiser  # noqa: E402
from seva.utils import load_model  # noqa: E402
from run_seva_keyframe_video import (  # noqa: E402
    build_target_trajectory_from_input_keyframes,
    preprocess_advanced,
)

VERSION_DICT = {"H": 576, "W": 576, "T": 21, "C": 4, "f": 8, "options": {}}


def find_two_closest(idx: int, c2ws: np.ndarray) -> tuple[int, int]:
    """Return indices of two closest cameras to c2ws[idx] by position."""
    pos = c2ws[idx, :3, 3]
    dists = [(i, np.linalg.norm(c2ws[i, :3, 3] - pos))
             for i in range(len(c2ws)) if i != idx]
    dists.sort(key=lambda x: x[1])
    r1 = dists[0][0]
    r2 = dists[1][0] if len(dists) > 1 else r1
    return r1, r2


def resolve_colmap_images(scene_dir: Path, image_names: list[str]):
    """Load COLMAP scene and resolve named images."""
    cameras, images = read_scene(scene_dir)
    images_dir = scene_dir / "images"
    by_name = {}
    for img in images.values():
        norm = img.name.replace("\\", "/")
        by_name[norm] = img
        by_name[Path(norm).name] = img
    resolved = []
    for name in image_names:
        img = by_name.get(name) or by_name.get(Path(name).name)
        if img is None:
            raise ValueError(f"Image '{name}' not found in COLMAP scene")
        resolved.append(img)
    return cameras, images_dir, resolved


def keyframe_indices_in_video(num_inputs: int, num_target_frames: int) -> list[int]:
    """Map input keyframes to their approximate frame indices in the output video.

    SEVA puts input frames first (indices 0..num_inputs-1), then target trajectory
    frames. The output video contains only the target trajectory frames.
    Keyframe i corresponds roughly to frame i*(num_target_frames-1)/(num_inputs-1).
    """
    if num_inputs <= 1:
        return [0]
    return [round(i * (num_target_frames - 1) / (num_inputs - 1))
            for i in range(num_inputs)]


def main():
    parser = argparse.ArgumentParser(description="Coupled SEVA + PoseSD inference")
    parser.add_argument("--scene", required=True, help="COLMAP scene directory")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Ordered image names from the scene")
    parser.add_argument("--checkpoint", required=True, help="PoseSD checkpoint")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--save-root", default="work_dirs/coupled")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--transition-sec", type=float, default=1.5)
    parser.add_argument("--pose-sd-cfg", type=float, default=3.0)
    parser.add_argument("--pose-sd-steps", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    scene_dir = Path(args.scene)
    image_names = args.inputs
    num_inputs = len(image_names)

    # --- Load COLMAP data for PoseSD (intrinsics + images) ---
    cameras, images_dir, colmap_imgs = resolve_colmap_images(scene_dir, image_names)
    img_paths = [str(images_dir / img.name) for img in colmap_imgs]

    # --- DUSt3R preprocessing (same as SEVA pipeline) ---
    print("Running DUSt3R...")
    dust3r = Dust3rPipeline(device=device)
    input_imgs, input_Ks, input_c2ws, (W, H) = preprocess_advanced(img_paths, dust3r)
    input_c2ws_np = input_c2ws.numpy()

    # --- Build trajectory ---
    target_c2ws, target_Ks = build_target_trajectory_from_input_keyframes(
        input_c2ws, input_Ks, (W, H), fps=args.fps, transition_sec=args.transition_sec,
    )
    num_targets = len(target_c2ws)
    print(f"Trajectory: {num_targets} frames from {num_inputs} keyframes")

    # --- Run SEVA ---
    print("Loading SEVA...")
    model = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
    ae = AutoEncoder(chunk_size=1).to(device)
    conditioner = CLIPConditioner().to(device)
    denoiser = DiscreteDenoiser(num_idx=1000, device=device)

    T = VERSION_DICT["T"]
    vd = copy.deepcopy(VERSION_DICT)
    num_anchors = infer_prior_stats(T, num_inputs, num_total_frames=num_targets, version_dict=vd)
    T = vd["T"]

    all_c2ws = torch.cat([input_c2ws, target_c2ws], 0)
    all_Ks = torch.cat([input_Ks, target_Ks], 0) * input_Ks.new_tensor([W, H, 1.0])[:, None]

    input_indices = list(range(num_inputs))
    anchor_indices = np.linspace(num_inputs, num_inputs + num_targets - 1, num_anchors).tolist()

    all_imgs_np = (
        F.pad(input_imgs, (0, 0, 0, 0, 0, 0, 0, num_targets), value=0.0).numpy() * 255.0
    ).astype(np.uint8)

    options = {
        "chunk_strategy": "interp-gt",
        "video_save_fps": args.fps,
        "beta_linear_start": 5e-6,
        "log_snr_shift": 2.4,
        "guider_types": [1, 2],
        "cfg": [3.0, 3.0 if num_inputs >= 9 else 2.0],
        "camera_scale": 2.0,
        "num_steps": 50,
        "cfg_min": 1.2,
        "encoding_t": 1,
        "decoding_t": 1,
    }

    render_dir = os.path.join(args.save_root, datetime.now().strftime("%Y%m%d_%H%M%S"))

    print("Running SEVA...")
    video_gen = run_one_scene(
        task="img2trajvid",
        version_dict={"H": H, "W": W, "T": T, "C": 4, "f": 8, "options": options},
        model=model,
        ae=ae,
        conditioner=conditioner,
        denoiser=denoiser,
        image_cond={"img": all_imgs_np, "input_indices": input_indices,
                     "prior_indices": anchor_indices},
        camera_cond={"c2w": all_c2ws, "K": all_Ks,
                      "input_indices": list(range(num_inputs + num_targets))},
        save_path=render_dir,
        use_traj_prior=True,
        traj_prior_c2ws=all_c2ws[[round(i) for i in anchor_indices]],
        traj_prior_Ks=all_Ks[[round(i) for i in anchor_indices]],
        seed=args.seed,
        gradio=True,
    )
    seva_video = None
    for vp in video_gen:
        seva_video = vp
    if seva_video is None:
        raise RuntimeError("SEVA produced no video")

    # Free SEVA from GPU
    del model, ae, conditioner, denoiser
    torch.cuda.empty_cache()

    # --- Read SEVA frames ---
    cap = cv2.VideoCapture(seva_video)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    num_video_frames = len(frames)
    print(f"SEVA output: {num_video_frames} frames")

    # --- Load PoseSD ---
    print("Loading PoseSD...")
    pose_sd = PoseSD()
    load_pose_sd_checkpoint(pose_sd, args.checkpoint, pose_sd.device)
    pose_sd.eval()

    # Prepare PoseSD inputs: COLMAP intrinsics + image tensors at 576x576
    H2d, W2d = 576, 576
    lh, lw = H2d // 8, W2d // 8
    colmap_Ks = [build_cropped_scaled_intrinsics(cameras[ci.camera_id], H2d, W2d)
                 for ci in colmap_imgs]
    ref_tensors = [load_image_tensor(images_dir, ci.name, H2d, W2d)[0].unsqueeze(0)
                   for ci in colmap_imgs]

    # Map each keyframe to its position in the video
    kf_video_indices = keyframe_indices_in_video(num_inputs, num_video_frames)

    # --- Replace keyframe positions with PoseSD outputs ---
    print("PoseSD canonicalization...")
    for input_idx in range(num_inputs):
        frame_idx = kf_video_indices[input_idx]
        if frame_idx >= num_video_frames:
            continue

        r1, r2 = find_two_closest(input_idx, input_c2ws_np)
        anchor_c2w = input_c2ws_np[r1]

        pl_r1 = compute_plucker_tensor(anchor_c2w, input_c2ws_np[r1],
                                        colmap_Ks[r1], H2d, W2d, lh, lw).unsqueeze(0)
        pl_r2 = compute_plucker_tensor(anchor_c2w, input_c2ws_np[r2],
                                        colmap_Ks[r2], H2d, W2d, lh, lw).unsqueeze(0)
        pl_tgt = compute_plucker_tensor(anchor_c2w, input_c2ws_np[input_idx],
                                         colmap_Ks[input_idx], H2d, W2d, lh, lw).unsqueeze(0)

        with torch.inference_mode():
            gen = pose_sd.sample(
                ref1_img=ref_tensors[r1], ref2_img=ref_tensors[r2],
                pl_ref1=pl_r1, pl_ref2=pl_r2, pl_tgt=pl_tgt,
                prompt="", num_steps=args.pose_sd_steps,
                cfg_scale=args.pose_sd_cfg, seed=args.seed,
            )[0]

        gen_np = ((gen.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
        gen_bgr = cv2.cvtColor(gen_np, cv2.COLOR_RGB2BGR)
        fh, fw = frames[frame_idx].shape[:2]
        if gen_bgr.shape[:2] != (fh, fw):
            gen_bgr = cv2.resize(gen_bgr, (fw, fh))
        frames[frame_idx] = gen_bgr
        print(f"  Frame {frame_idx}: {image_names[input_idx]} "
              f"(refs: {image_names[r1]}, {image_names[r2]})")

    # --- Write output ---
    out_path = args.output or os.path.join(render_dir, "coupled_output.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fh, fw = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              args.fps, (fw, fh))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"Saved: {out_path} ({num_video_frames} frames)")


if __name__ == "__main__":
    main()
