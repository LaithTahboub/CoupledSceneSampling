"""Inference utility for the single-reference debug model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from css.data.dataset import load_image_name_set
from css.debug_single_ref_experiment import SingleRefPairDataset, SingleRefPoseSD
from css.models.EMA import load_pose_sd_checkpoint


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_lines(path: str | None) -> list[str]:
    if path is None:
        return []
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    return ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()


def save_grid(
    *,
    ref_img: torch.Tensor,
    target_img: torch.Tensor,
    generated: list[torch.Tensor],
    output_path: Path,
    title: str,
) -> None:
    tiles = [_to_uint8(ref_img), _to_uint8(target_img)] + [_to_uint8(x) for x in generated]
    grid = np.concatenate(tiles, axis=1)
    canvas = Image.new("RGB", (grid.shape[1], grid.shape[0] + 30), color=(255, 255, 255))
    canvas.paste(Image.fromarray(grid), (0, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title[:260], fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--output", type=str, default="debug_single_ref_infer.png")

    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--scenes-file", type=str, default=None)
    p.add_argument("--pair-index", type=int, default=None, help="Index in mined pair dataset")
    p.add_argument("--prompt", type=str, default=None, help="Override prompt")
    p.add_argument("--prompt-template", type=str, default="a photo of {scene}")

    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--num-steps", type=int, default=40)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained-model", type=str, default="manojb/stable-diffusion-2-1-base")

    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--exclude-image-list", type=str, default=None)
    p.add_argument("--target-include-image-list", type=str, default=None)
    p.add_argument("--reference-include-image-list", type=str, default=None)

    p.add_argument("--min-pair-iou", type=float, default=0.18)
    p.add_argument("--max-pair-iou", type=float, default=0.62)
    p.add_argument("--min-pair-distance", type=float, default=0.20)
    p.add_argument("--max-pair-distance", type=float, default=2.2)
    p.add_argument("--min-view-cos", type=float, default=0.80)
    p.add_argument("--min-rotation-deg", type=float, default=3.0)
    p.add_argument("--max-rotation-deg", type=float, default=35.0)
    p.add_argument("--max-focal-ratio", type=float, default=1.35)
    p.add_argument("--prefilter-topk", type=int, default=64)
    p.add_argument("--targets-per-ref", type=int, default=2)
    p.add_argument("--max-pairs-per-scene", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    scenes = list(dict.fromkeys((args.scenes or []) + _read_lines(args.scenes_file)))
    if len(scenes) == 0:
        raise ValueError("Provide --scenes or --scenes-file")

    exclude_image_names = load_image_name_set(args.exclude_image_list)
    target_include_image_names = load_image_name_set(args.target_include_image_list)
    reference_include_image_names = load_image_name_set(args.reference_include_image_list)

    dataset = SingleRefPairDataset(
        scene_dirs=scenes,
        H=args.H,
        W=args.W,
        min_pair_iou=args.min_pair_iou,
        max_pair_iou=args.max_pair_iou,
        min_pair_distance=args.min_pair_distance,
        max_pair_distance=args.max_pair_distance,
        min_view_cos=args.min_view_cos,
        min_rotation_deg=args.min_rotation_deg,
        max_rotation_deg=args.max_rotation_deg,
        max_focal_ratio=args.max_focal_ratio,
        prefilter_topk=args.prefilter_topk,
        targets_per_ref=args.targets_per_ref,
        max_pairs_per_scene=args.max_pairs_per_scene,
        exclude_image_names=exclude_image_names,
        target_include_image_names=target_include_image_names,
        reference_include_image_names=reference_include_image_names,
        prompt_template=args.prompt_template,
    )
    if len(dataset) == 0:
        raise ValueError("No valid reference-target pairs found for inference.")

    if args.pair_index is None:
        idx = int(np.random.default_rng(args.seed).integers(0, len(dataset)))
    else:
        if args.pair_index < 0 or args.pair_index >= len(dataset):
            raise ValueError(f"pair-index {args.pair_index} out of range [0, {len(dataset) - 1}]")
        idx = int(args.pair_index)

    sample = dataset[idx]
    rec = dataset.pairs[idx]
    prompt = args.prompt if args.prompt is not None else sample["prompt"]

    print(f"Loading model from {args.checkpoint}")
    model = SingleRefPoseSD(pretrained_model=args.pretrained_model)
    load_pose_sd_checkpoint(model, args.checkpoint, model.device)
    model.eval()

    ref = sample["ref_img"].unsqueeze(0).to(model.device)
    generated: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(args.num_samples):
            out = model.sample(
                ref_img=ref,
                prompt=prompt,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed + i,
            )[0].detach().cpu()
            generated.append(out)

    output_path = Path(args.output)
    title = (
        f"{rec.scene_name} | ref={rec.ref_name} | tgt={rec.target_name} | "
        f"iou={rec.iou:.3f} | prompt={prompt}"
    )
    save_grid(
        ref_img=sample["ref_img"],
        target_img=sample["target_img"],
        generated=generated,
        output_path=output_path,
        title=title,
    )

    meta = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "pair_index": idx,
        "scene_name": rec.scene_name,
        "ref_name": rec.ref_name,
        "target_name": rec.target_name,
        "pair_iou": float(rec.iou),
        "pair_distance": float(rec.distance),
        "pair_rot_deg": float(rec.rot_deg),
        "prompt": prompt,
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "seed_base": args.seed,
        "output": str(output_path.resolve()),
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Selected pair idx={idx}: ref={rec.ref_name} target={rec.target_name} iou={rec.iou:.3f}")
    print(f"Saved grid: {output_path.resolve()}")
    print(f"Saved metadata: {meta_path.resolve()}")


if __name__ == "__main__":
    main()
