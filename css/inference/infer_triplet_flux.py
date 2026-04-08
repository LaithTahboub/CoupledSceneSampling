"""Triplet inference with RelightFlux.

Generates a 4-panel grid: ref1 | ref2 | gt | generated.
Supports either:
  --scene  : use COLMAP poses from a reconstructed scene directory
  (default): use MAST3R to estimate cameras from three input images
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

from css.data.dataset import compute_plucker_tensor
from css.inference.infer_triplet import adjust_K_for_crop_resize, load_image_tensor_exact
from css.inference.scene_sampling import build_comparison_grid, build_single_sample, load_scene_pools
from css.models.relight_flux import RelightFlux
from css.train.train_relight_flux import load_relight_flux_checkpoint

_SEVA_ROOT = str(Path(__file__).resolve().parent.parent.parent / "stable-virtual-camera")
if _SEVA_ROOT not in sys.path:
    sys.path.insert(0, _SEVA_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Triplet inference with MAST3R + RelightFlux")
    parser.add_argument("--ref1", required=True, help="Reference image 1 (path or name within scene)")
    parser.add_argument("--ref2", required=True, help="Reference image 2 (path or name within scene)")
    parser.add_argument("--target", required=True, help="Target image (path or name within scene)")
    parser.add_argument("--checkpoint", required=True, help="RelightFlux checkpoint")
    parser.add_argument("--scene", default=None, help="Scene directory with COLMAP data (skip MAST3R)")
    parser.add_argument("--output", default="output_flux.png")
    parser.add_argument("--prompt", default="", help="Text prompt for generation")
    parser.add_argument("--H", type=int, default=1024)
    parser.add_argument("--W", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="Geometry CFG scale")
    parser.add_argument("--cfg-text", type=float, default=3.0, help="Text CFG scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-crop", action="store_true",
                        help="Pad (letterbox) instead of center-crop to preserve full FoV")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    H, W = args.H, args.W
    latent_h, latent_w = H // 8, W // 8

    if args.scene:
        scene_dir = Path(args.scene)
        cameras, images_dir, all_images, _ = load_scene_pools(scene_dir)

        def find_image(query: str):
            query_path = Path(query)
            query_name = query_path.name
            try:
                query_rel = query_path.resolve().relative_to(images_dir.resolve()).as_posix()
            except ValueError:
                query_rel = None

            for img in all_images:
                if img.name == query or Path(img.name).name == query_name:
                    return img
                if query_rel and img.name == query_rel:
                    return img
            raise ValueError(f"Image '{query}' not found in scene")

        sample = build_single_sample(
            cameras,
            images_dir,
            find_image(args.ref1),
            find_image(args.ref2),
            find_image(args.target),
            H,
            W,
        )
    else:
        from seva.modules.preprocessor import get_pose_pipeline  # noqa: E402

        pipe = get_pose_pipeline("mast3r", device=args.device)
        imgs_np, Ks, c2ws, _, _ = pipe.infer_cameras_and_points([args.ref1, args.ref2, args.target])
        orig_sizes = [(img.shape[1], img.shape[0]) for img in imgs_np]
        del pipe
        torch.cuda.empty_cache()

        K_ref1, K_ref2, K_tgt = [
            adjust_K_for_crop_resize(K, src_w, src_h, H, W, pad=args.no_crop)
            for K, (src_w, src_h) in zip(Ks, orig_sizes, strict=True)
        ]
        c2w_ref1, c2w_ref2, c2w_tgt = c2ws

        sample = {
            "ref1_img": load_image_tensor_exact(Path(args.ref1), H, W, pad=args.no_crop),
            "ref2_img": load_image_tensor_exact(Path(args.ref2), H, W, pad=args.no_crop),
            "target_img": load_image_tensor_exact(Path(args.target), H, W, pad=args.no_crop),
            "plucker_ref1": compute_plucker_tensor(
                c2w_ref1, c2w_ref1, K_ref1, H, W, latent_h, latent_w,
            ).unsqueeze(0),
            "plucker_ref2": compute_plucker_tensor(
                c2w_ref1, c2w_ref2, K_ref2, H, W, latent_h, latent_w,
            ).unsqueeze(0),
            "plucker_tgt": compute_plucker_tensor(
                c2w_ref1, c2w_tgt, K_tgt, H, W, latent_h, latent_w,
            ).unsqueeze(0),
        }

    model = RelightFlux(device=args.device)
    load_relight_flux_checkpoint(model, args.checkpoint, model.device)
    model.eval()

    with torch.inference_mode():
        generated = model.sample(
            ref1_img=sample["ref1_img"],
            ref2_img=sample["ref2_img"],
            pl_ref1=sample["plucker_ref1"],
            pl_ref2=sample["plucker_ref2"],
            pl_tgt=sample["plucker_tgt"],
            prompt=args.prompt,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            cfg_text=args.cfg_text,
            seed=args.seed,
        )

    grid = build_comparison_grid(
        sample["ref1_img"][0],
        sample["ref2_img"][0],
        sample["target_img"][0],
        generated[0],
        prompt=args.prompt or None,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)
    print(f"Saved: {output_path}  [Ref1 | Ref2 | GT | Generated]")


if __name__ == "__main__":
    main()
