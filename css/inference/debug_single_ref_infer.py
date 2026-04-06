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
from css.models.EMA import load_relight_sd_checkpoint


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
    *, ref_img: torch.Tensor, target_img: torch.Tensor | None,
    generated: list[torch.Tensor], output_path: Path, title: str | None,
) -> None:
    images = [_to_uint8(ref_img)]
    if target_img is not None:
        images.append(_to_uint8(target_img))
    
    tiles = images + [_to_uint8(x) for x in generated]
    grid = np.concatenate(tiles, axis=1)
    
    offset = 30 if title else 0
    canvas = Image.new("RGB", (grid.shape[1], grid.shape[0] + offset), color=(255, 255, 255))
    canvas.paste(Image.fromarray(grid), (0, offset))
    
    if title:
        ImageDraw.Draw(canvas).text((8, 8), title[:260], fill=(0, 0, 0))
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/vulcanscratch/ltahboub/CoupledSceneSampling/checkpoints/single_ref_debug/unet_final.pt", type=str)
    p.add_argument("--output", type=str, default="debug_single_ref_infer.png")

    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--scenes-file", type=str, default=None)
    p.add_argument("--pair-index", type=int, default=None)
    p.add_argument("--prompt", type=str, default=None)

    p.add_argument("--ref-image", type=str, default=None)
    
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=40)
    p.add_argument("--cfg-scale", type=float, default=4)
    p.add_argument("--seed", type=int, default=51)
    p.add_argument("--pretrained-model", type=str, default="manojb/stable-diffusion-2-1-base")

    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--min-covisibility", type=float, default=0.20)
    p.add_argument("--max-covisibility", type=float, default=0.80)
    p.add_argument("--max-pairs-per-scene", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    if args.ref_image:
        raw_img = Image.open(args.ref_image).convert("RGB").resize((args.W, args.H))
        ref_img = (torch.from_numpy(np.array(raw_img)).permute(2, 0, 1).float() / 127.5) - 1.0
        target_img = None
        prompt = args.prompt or ""
        title = None
    else:
        scenes = list(dict.fromkeys((args.scenes or []) + _read_lines(args.scenes_file)))
        if not scenes:
            raise ValueError("Provide --scenes, --scenes-file, or --ref-image")

        dataset = SingleRefPairDataset(
            scene_dirs=scenes, H=args.H, W=args.W,
            min_covisibility=args.min_covisibility, max_covisibility=args.max_covisibility,
            max_pairs_per_scene=args.max_pairs_per_scene,
        )
        idx = args.pair_index if args.pair_index is not None else random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        rec = dataset.pairs[idx]
        ref_img, target_img = sample["ref_img"], sample["target_img"]
        prompt = args.prompt or sample["prompt"]
        title = f"{rec.scene_name} | ref={rec.ref_name} | tgt={rec.target_name} | covis={rec.covisibility:.3f}"

    model = SingleRefSD(pretrained_model=args.pretrained_model)
    load_relight_sd_checkpoint(model, args.checkpoint, model.device)
    model.eval()

    generated: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(args.num_samples):
            out = model.sample(ref_img=ref_img.unsqueeze(0), prompt=prompt,
                               num_steps=args.num_steps, cfg_scale=args.cfg_scale,
                               seed=args.seed + i)[0].cpu()
            generated.append(out)

    output_path = Path(args.output)
    save_grid(ref_img=ref_img, target_img=target_img, generated=generated, 
              output_path=output_path, title=title)
    print(f"Saved: {output_path.resolve()}")

if __name__ == "__main__":
    main()
