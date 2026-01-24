import glob
import os
import os.path as osp
import gc  # Added for memory management

import fire
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from seva.data_io import get_parser
from seva.eval import (
    IS_TORCH_NIGHTLY,
    compute_relative_inds,
    create_transforms_simple,
    infer_prior_inds,
    infer_prior_stats,
    run_one_scene,
)
from seva.geometry import (
    generate_interpolated_path,
    generate_spiral_path,
    get_arc_horizontal_w2cs,
    get_default_intrinsics,
    get_lookat,
    get_preset_pose_fov,
)
from seva.model import SGMWrapper
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner
from seva.sampling import DiscreteDenoiser
from seva.utils import load_model

device = "cuda:0"


# Constants.
WORK_DIR = "work_dirs/demo"

# --- MODIFICATION: Default to False to save memory unless strictly nightly ---
# Compilation can consume extra VRAM for graph caching.
if IS_TORCH_NIGHTLY:
    COMPILE = True
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
else:
    COMPILE = False

# Initialize lightweight modules. 
# Note: We initialize them, but we will ensure memory is clean before main loop.
AE = AutoEncoder(chunk_size=1).to(device)
CONDITIONER = CLIPConditioner().to(device)
DENOISER = DiscreteDenoiser(num_idx=1000, device=device)

if COMPILE:
    print("Compiling helper models...")
    CONDITIONER = torch.compile(CONDITIONER, dynamic=False)
    AE = torch.compile(AE, dynamic=False)


def parse_task(
    task,
    scene,
    num_inputs,
    T,
    version_dict,
):
    options = version_dict["options"]

    anchor_indices = None
    anchor_c2ws = None
    anchor_Ks = None

    if task == "img2trajvid_s-prob":
        if num_inputs is not None:
            assert (
                num_inputs == 1
            ), "Task `img2trajvid_s-prob` only support 1-view conditioning..."
        else:
            num_inputs = 1
        
        num_targets = options.get("num_targets", T - 1)
        
        # --- FIX: Sync global T with requested targets to prevent index errors ---
        version_dict["T"] = num_inputs + num_targets
        T = version_dict["T"] 
        # -----------------------------------------------------------------------

        num_anchors = infer_prior_stats(
            T,
            num_inputs,
            num_total_frames=num_targets,
            version_dict=version_dict,
        )

        input_indices = [0]
        anchor_indices = np.linspace(1, num_targets, num_anchors).tolist()

        all_imgs_path = [scene] + [None] * num_targets

        # --- MODIFICATION: Custom Circle Trajectory Support ---
        requested_prior = options.get("traj_prior", "orbit")
        # Pass 'orbit' to library function to get safe defaults
        safe_prior = requested_prior if requested_prior not in ["circle"] else "orbit"

        c2ws, fovs = get_preset_pose_fov(
            option=safe_prior,
            num_frames=num_targets + 1,
            start_w2c=torch.eye(4),
            look_at=torch.Tensor([0, 0, 10]),
        )

        if requested_prior == "circle":
            radius = 10.0
            num_frames = c2ws.shape[0]
            # Trajectory: Quarter circle to the right (-pi/2)
            angles = torch.linspace(0, -torch.pi / 2, num_frames)

            new_c2ws = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)
            s = torch.sin(angles)
            c = torch.cos(angles)

            # Rotation (Y-axis)
            new_c2ws[:, 0, 0] = c
            new_c2ws[:, 0, 2] = -s
            new_c2ws[:, 2, 0] = s
            new_c2ws[:, 2, 2] = c

            # Translation
            new_c2ws[:, 0, 3] = -radius * s
            new_c2ws[:, 2, 3] = radius * (1 - c)

            c2ws = new_c2ws
        # ------------------------------------------------------

        with Image.open(scene) as img:
            W, H = img.size
            aspect_ratio = W / H
        Ks = get_default_intrinsics(fovs, aspect_ratio=aspect_ratio)  # unormalized
        Ks[:, :2] *= (
            torch.tensor([W, H]).reshape(1, -1, 1).repeat(Ks.shape[0], 1, 1)
        )  # normalized
        Ks = Ks.numpy()

        anchor_c2ws = c2ws[[round(ind) for ind in anchor_indices]]
        anchor_Ks = Ks[[round(ind) for ind in anchor_indices]]

    else:
        # (Standard logic for other tasks remains unchanged)
        parser = get_parser(
            parser_type="reconfusion",
            data_dir=scene,
            normalize=False,
        )
        all_imgs_path = parser.image_paths
        c2ws = parser.camtoworlds
        camera_ids = parser.camera_ids
        Ks = np.concatenate([parser.Ks_dict[cam_id][None] for cam_id in camera_ids], 0)

        if num_inputs is None:
            assert len(parser.splits_per_num_input_frames.keys()) == 1
            num_inputs = list(parser.splits_per_num_input_frames.keys())[0]
            split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore
        elif isinstance(num_inputs, str):
            split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore
            num_inputs = int(num_inputs.split("-")[0])  # for example 1_from32
        else:
            split_dict = parser.splits_per_num_input_frames[num_inputs]  # type: ignore

        num_targets = len(split_dict["test_ids"])

        if task == "img2img":
            num_anchors = infer_prior_stats(
                T,
                num_inputs,
                num_total_frames=num_targets,
                version_dict=version_dict,
            )

            sampled_indices = np.sort(
                np.array(split_dict["train_ids"] + split_dict["test_ids"])
            ) 

            traj_prior = options.get("traj_prior", None)
            if traj_prior == "spiral":
                assert parser.bounds is not None
                anchor_c2ws = generate_spiral_path(
                    c2ws[sampled_indices] @ np.diagflat([1, -1, -1, 1]),
                    parser.bounds[sampled_indices],
                    n_frames=num_anchors + 1,
                    n_rots=2,
                    zrate=0.5,
                    endpoint=False,
                )[1:] @ np.diagflat([1, -1, -1, 1])
            elif traj_prior == "interpolated":
                assert num_inputs > 1
                anchor_c2ws = generate_interpolated_path(
                    c2ws[split_dict["train_ids"], :3],
                    round((num_anchors + 1) / (num_inputs - 1)),
                    endpoint=False,
                )[1 : num_anchors + 1]
            elif traj_prior == "orbit":
                c2ws_th = torch.as_tensor(c2ws)
                lookat = get_lookat(
                    c2ws_th[sampled_indices, :3, 3],
                    c2ws_th[sampled_indices, :3, 2],
                )
                anchor_c2ws = torch.linalg.inv(
                    get_arc_horizontal_w2cs(
                        torch.linalg.inv(c2ws_th[split_dict["train_ids"][0]]),
                        lookat,
                        -F.normalize(
                            c2ws_th[split_dict["train_ids"]][:, :3, 1].mean(0),
                            dim=-1,
                        ),
                        num_frames=num_anchors + 1,
                        endpoint=False,
                    )
                ).numpy()[1:, :3]
            else:
                anchor_c2ws = None

            all_imgs_path = [all_imgs_path[i] for i in sampled_indices]
            c2ws = c2ws[sampled_indices]
            Ks = Ks[sampled_indices]

            input_indices = compute_relative_inds(
                sampled_indices,
                np.array(split_dict["train_ids"]),
            )
            anchor_indices = np.arange(
                sampled_indices.shape[0],
                sampled_indices.shape[0] + num_anchors,
            ).tolist()

        elif task == "img2vid":
            num_targets = len(all_imgs_path) - num_inputs
            num_anchors = infer_prior_stats(
                T,
                num_inputs,
                num_total_frames=num_targets,
                version_dict=version_dict,
            )

            input_indices = split_dict["train_ids"]
            anchor_indices = infer_prior_inds(
                c2ws,
                num_prior_frames=num_anchors,
                input_frame_indices=input_indices,
                options=options,
            ).tolist()
            num_anchors = len(anchor_indices)
            anchor_c2ws = c2ws[anchor_indices, :3]
            anchor_Ks = Ks[anchor_indices]

        elif task == "img2trajvid":
            num_anchors = infer_prior_stats(
                T,
                num_inputs,
                num_total_frames=num_targets,
                version_dict=version_dict,
            )

            target_c2ws = c2ws[split_dict["test_ids"], :3]
            target_Ks = Ks[split_dict["test_ids"]]
            anchor_c2ws = target_c2ws[
                np.linspace(0, num_targets - 1, num_anchors).round().astype(np.int64)
            ]
            anchor_Ks = target_Ks[
                np.linspace(0, num_targets - 1, num_anchors).round().astype(np.int64)
            ]

            sampled_indices = split_dict["train_ids"] + split_dict["test_ids"]
            all_imgs_path = [all_imgs_path[i] for i in sampled_indices]
            c2ws = c2ws[sampled_indices]
            Ks = Ks[sampled_indices]

            input_indices = np.arange(num_inputs).tolist()
            anchor_indices = np.linspace(
                num_inputs, num_inputs + num_targets - 1, num_anchors
            ).tolist()

        else:
            raise ValueError(f"Unknown task: {task}")

    return (
        all_imgs_path,
        num_inputs,
        num_targets,
        input_indices,
        anchor_indices,
        torch.tensor(c2ws[:, :3]).float(),
        torch.tensor(Ks).float(),
        (torch.tensor(anchor_c2ws[:, :3]).float() if anchor_c2ws is not None else None),
        (torch.tensor(anchor_Ks).float() if anchor_Ks is not None else None),
    )


def main(
    data_path,
    data_items=None,
    version=1.1,
    task="img2img",
    save_subdir="",
    H=None,
    W=None,
    T=None,
    use_traj_prior=False,
    pretrained_model_name_or_path="stabilityai/stable-virtual-camera",
    weight_name="model.safetensors",
    seed=23,
    **overwrite_options,
):
    # --- MODIFICATION: Clear cache before loading model ---
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Loading Model (Target T={T})...")
    MODEL = SGMWrapper(
        load_model(
            model_version=version,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            weight_name=weight_name,
            device="cpu",
            verbose=True,
        ).eval()
    ).to(device)

    if COMPILE:
        MODEL = torch.compile(MODEL, dynamic=False)

    VERSION_DICT = {
        "H": H or 576,
        "W": W or 576,
        "T": ([int(t) for t in T.split(",")] if isinstance(T, str) else T) or 21,
        "C": 4,
        "f": 8,
        "options": {
            "chunk_strategy": "nearest-gt",
            "video_save_fps": 30.0,
            "beta_linear_start": 5e-6,
            "log_snr_shift": 2.4,
            "guider_types": 1,
            "cfg": 2.0,
            "camera_scale": 2.0,
            "num_steps": 50,
            "cfg_min": 1.2,
            "encoding_t": 1,
            "decoding_t": 1,
        },
    }

    options = VERSION_DICT["options"]
    options.update(overwrite_options)

    if data_items is not None:
        if not isinstance(data_items, (list, tuple)):
            data_items = data_items.split(",")
        scenes = [os.path.join(data_path, item) for item in data_items]
    else:
        scenes = [
            item for item in glob.glob(osp.join(data_path, "*")) if os.path.isfile(item)
        ]

    for scene in tqdm(scenes):
        num_inputs = options.get("num_inputs", None)
        save_path_scene = os.path.join(
            WORK_DIR, task, save_subdir, os.path.splitext(os.path.basename(scene))[0]
        )
        if options.get("skip_saved", False) and os.path.exists(
            os.path.join(save_path_scene, "transforms.json")
        ):
            print(f"Skipping {scene} as it is already sampled.")
            continue

        (
            all_imgs_path,
            num_inputs,
            num_targets,
            input_indices,
            anchor_indices,
            c2ws,
            Ks,
            anchor_c2ws,
            anchor_Ks,
        ) = parse_task(
            task,
            scene,
            num_inputs,
            VERSION_DICT["T"],
            VERSION_DICT,
        )
        assert num_inputs is not None
        
        image_cond = {
            "img": all_imgs_path,
            "input_indices": input_indices,
            "prior_indices": anchor_indices,
        }
        
        camera_cond = {
            "c2w": c2ws.clone(),
            "K": Ks.clone(),
            "input_indices": list(range(num_inputs + num_targets)),
        }
        
        # --- MODIFICATION: Wrap generation in no_grad and clear cache ---
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Generating {num_targets} frames (Total T={num_inputs + num_targets})...")
        with torch.no_grad():
            video_path_generator = run_one_scene(
                task,
                VERSION_DICT,
                model=MODEL,
                ae=AE,
                conditioner=CONDITIONER,
                denoiser=DENOISER,
                image_cond=image_cond,
                camera_cond=camera_cond,
                save_path=save_path_scene,
                use_traj_prior=use_traj_prior,
                traj_prior_Ks=anchor_Ks,
                traj_prior_c2ws=anchor_c2ws,
                seed=seed,
            )
            for _ in video_path_generator:
                pass
        # ---------------------------------------------------------------

        c2ws = c2ws @ torch.tensor(np.diag([1, -1, -1, 1])).float()
        img_paths = sorted(glob.glob(osp.join(save_path_scene, "samples-rgb", "*.png")))
        if len(img_paths) != len(c2ws):
            input_img_paths = sorted(
                glob.glob(osp.join(save_path_scene, "input", "*.png"))
            )
            assert len(img_paths) == num_targets
            assert len(input_img_paths) == num_inputs
            assert c2ws.shape[0] == num_inputs + num_targets
            target_indices = [i for i in range(c2ws.shape[0]) if i not in input_indices]
            img_paths = [
                input_img_paths[input_indices.index(i)]
                if i in input_indices
                else img_paths[target_indices.index(i)]
                for i in range(c2ws.shape[0])
            ]
        create_transforms_simple(
            save_path=save_path_scene,
            img_paths=img_paths,
            img_whs=np.array([VERSION_DICT["W"], VERSION_DICT["H"]])[None].repeat(
                num_inputs + num_targets, 0
            ),
            c2ws=c2ws,
            Ks=Ks,
        )


if __name__ == "__main__":
    fire.Fire(main)