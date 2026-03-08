import argparse
import copy
import os
import os.path as osp
import shutil
from datetime import datetime

import numpy as np
import scipy.interpolate
import splines
import splines.quaternion
import torch
import torch.nn.functional as F
import viser.transforms as vt
from einops import rearrange

from seva.eval import infer_prior_stats, run_one_scene, transform_img_and_K
from seva.geometry import normalize_scene
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.modules.preprocessor import Dust3rPipeline
from seva.sampling import DiscreteDenoiser
from seva.utils import load_model


VERSION_DICT = {
    "H": 576,
    "W": 576,
    "T": 21,
    "C": 4,
    "f": 8,
    "options": {},
}


def preprocess_advanced(
    img_paths: list[str], dust3r: Dust3rPipeline
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    shorter = round(576 / 64) * 64
    input_imgs, input_Ks, input_c2ws, points, _ = dust3r.infer_cameras_and_points(
        img_paths
    )
    num_inputs = len(img_paths)
    if num_inputs == 1:
        input_imgs, input_Ks, input_c2ws, points = (
            input_imgs[:1],
            input_Ks[:1],
            input_c2ws[:1],
            points[:1],
        )

    input_imgs = [img[..., :3] for img in input_imgs]

    point_chunks = [p.shape[0] for p in points]
    point_indices = np.cumsum(point_chunks)[:-1]
    input_c2ws, points, _ = normalize_scene(
        input_c2ws,
        np.concatenate(points, 0),
        camera_center_method="poses",
    )
    points = np.split(points, point_indices, 0)
    scene_scale = np.median(np.ptp(np.concatenate([input_c2ws[:, :3, 3], *points], 0), -1))
    input_c2ws[:, :3, 3] /= scene_scale

    input_imgs = [torch.as_tensor(img / 255.0, dtype=torch.float32) for img in input_imgs]
    input_Ks = torch.as_tensor(input_Ks, dtype=torch.float32)
    input_c2ws = torch.as_tensor(input_c2ws, dtype=torch.float32)

    resized_imgs = []
    resized_Ks = []
    for img, K in zip(input_imgs, input_Ks):
        img = rearrange(img, "h w c -> 1 c h w")
        img, K = transform_img_and_K(img, shorter, K=K[None], size_stride=64)
        assert isinstance(K, torch.Tensor)
        K = K / K.new_tensor([img.shape[-1], img.shape[-2], 1])[:, None]
        resized_imgs.append(img)
        resized_Ks.append(K)

    input_imgs = torch.cat(resized_imgs, 0)
    input_imgs = rearrange(input_imgs, "b c h w -> b h w c")[..., :3]
    input_Ks = torch.cat(resized_Ks, 0)
    input_wh = (input_imgs.shape[2], input_imgs.shape[1])
    return input_imgs, input_Ks, input_c2ws, input_wh


def get_default_fov_rad(input_Ks: torch.Tensor, input_wh: tuple[int, int]) -> float:
    W, H = input_wh
    if H > W:
        return float(2 * np.arctan(1.0 / (2 * input_Ks[0, 0, 0].item())))
    return float(2 * np.arctan(1.0 / (2 * input_Ks[0, 1, 1].item())))


def get_intrinsics(W: int, H: int, fov_rad: float) -> np.ndarray:
    focal = 0.5 * H / np.tan(0.5 * fov_rad)
    return np.array(
        [[focal, 0.0, 0.5 * W], [0.0, focal, 0.5 * H], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def build_target_trajectory_from_input_keyframes(
    input_c2ws: torch.Tensor,
    input_Ks: torch.Tensor,
    input_wh: tuple[int, int],
    fps: float = 30.0,
    transition_sec: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_keyframes = input_c2ws.shape[0]
    if num_keyframes < 2:
        raise ValueError("Need at least two input images to create keyframe trajectory.")

    duration = transition_sec * (num_keyframes - 1)
    num_frames = int(fps * duration)
    if num_frames <= 0:
        raise ValueError("Computed zero target frames; check fps/transition settings.")

    W, H = input_wh
    default_fov = get_default_fov_rad(input_Ks, input_wh)

    transition_times_cumsum = np.array(
        [i * transition_sec for i in range(num_keyframes)],
        dtype=np.float64,
    )
    spline_indices = np.arange(num_keyframes, dtype=np.float64)
    time_to_spline = scipy.interpolate.PchipInterpolator(
        x=transition_times_cumsum,
        y=spline_indices,
    )

    orientation_spline = splines.quaternion.KochanekBartels(
        [
            splines.quaternion.UnitQuaternion.from_unit_xyzw(
                np.roll(vt.SO3.from_matrix(c2w[:3, :3].numpy()).wxyz, shift=-1)
            )
            for c2w in input_c2ws
        ],
        tcb=(0.0, 0.0, 0.0),
        endconditions="natural",
    )
    position_spline = splines.KochanekBartels(
        [c2w[:3, 3].numpy() for c2w in input_c2ws],
        tcb=(0.0, 0.0, 0.0),
        endconditions="natural",
    )
    fov_spline = splines.KochanekBartels(
        [default_fov] * num_keyframes,
        tcb=(0.0, 0.0, 0.0),
        endconditions="natural",
    )

    max_t = transition_times_cumsum[-1]
    target_c2ws = []
    target_Ks = []
    for i in range(num_frames):
        t = max_t * (i / num_frames)
        spline_t = float(np.clip(time_to_spline(t), 0.0, spline_indices[-1]))

        quat = orientation_spline.evaluate(spline_t)
        position = np.asarray(position_spline.evaluate(spline_t), dtype=np.float32)
        fov_rad = float(fov_spline.evaluate(spline_t))

        wxyz = np.array([quat.scalar, *quat.vector], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = vt.SO3(wxyz).as_matrix().astype(np.float32)
        c2w[:3, 3] = position

        K = get_intrinsics(W, H, fov_rad)
        K = K / np.array([W, H, 1.0], dtype=np.float32)[:, None]

        target_c2ws.append(c2w)
        target_Ks.append(K)

    return (
        torch.as_tensor(np.stack(target_c2ws), dtype=torch.float32),
        torch.as_tensor(np.stack(target_Ks), dtype=torch.float32),
    )


def run(
    img_paths: list[str],
    output_path: str | None,
    save_root: str,
    seed: int,
    device: str,
) -> str:
    if len(img_paths) < 2:
        raise ValueError("Please pass at least two ordered images.")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA is not available; SEVA demo pipeline expects a CUDA device.")

    dust3r = Dust3rPipeline(device=device)
    model = SGMWrapper(load_model(device="cpu", verbose=True).eval()).to(device)
    ae = AutoEncoder(chunk_size=1).to(device)
    conditioner = CLIPConditioner().to(device)
    denoiser = DiscreteDenoiser(num_idx=1000, device=device)

    input_imgs, input_Ks, input_c2ws, (W, H) = preprocess_advanced(img_paths, dust3r)
    target_c2ws, target_Ks = build_target_trajectory_from_input_keyframes(
        input_c2ws=input_c2ws,
        input_Ks=input_Ks,
        input_wh=(W, H),
        fps=30.0,
        transition_sec=1.5,
    )

    num_inputs = len(input_imgs)
    num_targets = len(target_c2ws)
    input_indices = list(range(num_inputs))

    all_c2ws = torch.cat([input_c2ws, target_c2ws], 0)
    all_Ks = torch.cat([input_Ks, target_Ks], 0) * input_Ks.new_tensor([W, H, 1.0])[:, None]

    T = VERSION_DICT["T"]
    version_dict = copy.deepcopy(VERSION_DICT)
    num_anchors = infer_prior_stats(
        T,
        num_inputs,
        num_total_frames=num_targets,
        version_dict=version_dict,
    )
    assert isinstance(num_anchors, int)
    T = version_dict["T"]

    anchor_indices = np.linspace(
        num_inputs,
        num_inputs + num_targets - 1,
        num_anchors,
    ).tolist()
    anchor_c2ws = all_c2ws[[round(ind) for ind in anchor_indices]]
    anchor_Ks = all_Ks[[round(ind) for ind in anchor_indices]]

    all_imgs_np = (F.pad(input_imgs, (0, 0, 0, 0, 0, 0, 0, num_targets), value=0.0).numpy() * 255.0).astype(
        np.uint8
    )
    image_cond = {
        "img": all_imgs_np,
        "input_indices": input_indices,
        "prior_indices": anchor_indices,
    }
    camera_cond = {
        "c2w": all_c2ws,
        "K": all_Ks,
        "input_indices": list(range(num_inputs + num_targets)),
    }

    options = copy.deepcopy(VERSION_DICT["options"])
    options["chunk_strategy"] = "interp-gt"
    options["video_save_fps"] = 30.0
    options["beta_linear_start"] = 5e-6
    options["log_snr_shift"] = 2.4
    options["guider_types"] = [1, 2]
    options["cfg"] = [3.0, 3.0 if num_inputs >= 9 else 2.0]
    options["camera_scale"] = 2.0
    options["num_steps"] = 50
    options["cfg_min"] = 1.2
    options["encoding_t"] = 1
    options["decoding_t"] = 1

    render_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    render_dir = osp.join(save_root, render_name)

    video_path_generator = run_one_scene(
        task="img2trajvid",
        version_dict={
            "H": H,
            "W": W,
            "T": T,
            "C": VERSION_DICT["C"],
            "f": VERSION_DICT["f"],
            "options": options,
        },
        model=model,
        ae=ae,
        conditioner=conditioner,
        denoiser=denoiser,
        image_cond=image_cond,
        camera_cond=camera_cond,
        save_path=render_dir,
        use_traj_prior=True,
        traj_prior_c2ws=anchor_c2ws,
        traj_prior_Ks=anchor_Ks,
        seed=seed,
        gradio=True,
    )

    final_video_path = None
    for video_path in video_path_generator:
        final_video_path = video_path
    if final_video_path is None:
        raise RuntimeError("SEVA sampling did not produce a video path.")

    if output_path is None:
        return final_video_path

    os.makedirs(osp.dirname(osp.abspath(output_path)), exist_ok=True)
    shutil.copy2(final_video_path, output_path)
    return osp.abspath(output_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run SEVA on ordered input images using the same Advanced demo path: "
            "preprocess with DUSt3R, keyframe each image in order, chunk_strategy=interp-gt, "
            "camera_scale=2, cfg=3."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Ordered input image paths.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional final mp4 output path. If omitted, prints the generated path in work_dirs."
        ),
    )
    parser.add_argument(
        "--save-root",
        default="work_dirs/seva_keyframe_video",
        help="Directory for run artifacts.",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    final_video = run(
        img_paths=[osp.abspath(p) for p in args.inputs],
        output_path=args.output,
        save_root=args.save_root,
        seed=args.seed,
        device=args.device,
    )
    print(final_video)


if __name__ == "__main__":
    main()
