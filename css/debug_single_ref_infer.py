"""Inference utility for the single-reference debug model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from css.debug_single_ref_experiment import SingleRefPairDataset, SingleRefSD, _to_uint8
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
    lines: list[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def save_grid(
    *, ref_img: torch.Tensor, target_img: torch.Tensor,
    generated: list[torch.Tensor], output_path: Path, title: str,
) -> None:
    tiles = [_to_uint8(ref_img), _to_uint8(target_img)] + [_to_uint8(x) for x in generated]
    grid = np.concatenate(tiles, axis=1)
    canvas = Image.new("RGB", (grid.shape[1], grid.shape[0] + 30), color=(255, 255, 255))
    canvas.paste(Image.fromarray(grid), (0, 30))
    ImageDraw.Draw(canvas).text((8, 8), title[:260], fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--output", type=str, default="debug_single_ref_infer.png")

    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--scenes-file", type=str, default=None)
    p.add_argument("--pair-index", type=int, default=None)
    p.add_argument("--prompt", type=str, default=None)

    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--num-steps", type=int, default=40)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained-model", type=str, default="manojb/stable-diffusion-2-1-base")

    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--min-covisibility", type=float, default=0.10)
    p.add_argument("--max-covisibility", type=float, default=0.80)
    p.add_argument("--max-pairs-per-scene", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    scenes = list(dict.fromkeys((args.scenes or []) + _read_lines(args.scenes_file)))
    if not scenes:
        raise ValueError("Provide --scenes or --scenes-file")

    dataset = SingleRefPairDataset(
        scene_dirs=scenes, H=args.H, W=args.W,
        min_covisibility=args.min_covisibility, max_covisibility=args.max_covisibility,
        max_pairs_per_scene=args.max_pairs_per_scene,
    )
    if len(dataset) == 0:
        raise ValueError("No valid pairs found.")

    if args.pair_index is not None:
        idx = args.pair_index
    else:
        idx = int(np.random.default_rng(args.seed).integers(0, len(dataset)))

    sample = dataset[idx]
    rec = dataset.pairs[idx]
    prompt = args.prompt or sample["prompt"]

    print(f"Loading model from {args.checkpoint}")
    model = SingleRefSD(pretrained_model=args.pretrained_model)
    load_pose_sd_checkpoint(model, args.checkpoint, model.device)
    model.eval()

    ref = sample["ref_img"].unsqueeze(0)
    generated: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(args.num_samples):
            out = model.sample(ref_img=ref, prompt=prompt,
                               num_steps=args.num_steps, cfg_scale=args.cfg_scale,
                               seed=args.seed + i)[0].cpu()
            generated.append(out)

    output_path = Path(args.output)
    title = f"{rec.scene_name} | ref={rec.ref_name} | tgt={rec.target_name} | covis={rec.covisibility:.3f}"
    save_grid(ref_img=sample["ref_img"], target_img=sample["target_img"],
              generated=generated, output_path=output_path, title=title)

    meta = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "pair_index": idx, "scene_name": rec.scene_name,
        "ref_name": rec.ref_name, "target_name": rec.target_name,
        "covisibility": rec.covisibility, "prompt": prompt,
        "num_samples": args.num_samples, "seed_base": args.seed,
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
