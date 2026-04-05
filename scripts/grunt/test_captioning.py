"""Quick visual test of the VLM captioning pipeline.

For each scene dir provided, picks a random target image, captions it,
and saves a labeled copy to the output folder.

Usage:
    # Start vLLM server first, then:
    python test_captioning.py \
        --scenes /path/to/scene1 /path/to/scene2 \
        --output-dir test_captions_out \
        --api-base http://localhost:8000/v1
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from css.data.caption_dataset import caption_single_image, discover_scene_images


def draw_caption(img: Image.Image, caption: str, font_size: int = 16) -> Image.Image:
    """Return a copy of img with the caption rendered at the bottom."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Wrap text to fit image width
    max_w = img.width - 20
    words = caption.split()
    lines, cur = [], ""
    for w in words:
        test = f"{cur} {w}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_w and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)

    line_h = font_size + 4
    block_h = line_h * len(lines) + 16
    # Expand image to fit caption below
    new_img = Image.new("RGB", (img.width, img.height + block_h), (0, 0, 0))
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    y = img.height + 8
    for line in lines:
        draw.text((10, y), line, fill=(255, 255, 255), font=font)
        y += line_h
    return new_img


_DEFAULT_MEGASCENES_ROOT = "/fs/nexus-scratch/ltahboub/MegaScenes"


def _pick_random_scenes(megascenes_root: str, n: int) -> list[Path]:
    """Pick n random scene directories that contain an images/ subfolder."""
    root = Path(megascenes_root)
    candidates = [
        d for d in sorted(root.iterdir())
        if d.is_dir() and (d / "images").is_dir()
    ]
    if not candidates:
        raise RuntimeError(f"No scenes found in {root}")
    return random.sample(candidates, min(n, len(candidates)))


def main():
    p = argparse.ArgumentParser(description="Test VLM captioning on random scene targets")
    p.add_argument("--scenes", nargs="+", default=None, help="Scene directories (omit to use --num-scenes random)")
    p.add_argument("--num-scenes", type=int, default=5, help="Number of random scenes to pick (when --scenes is omitted)")
    p.add_argument("--num-images", type=int, default=1, help="Number of random images per scene")
    p.add_argument("--megascenes-root", type=str, default=_DEFAULT_MEGASCENES_ROOT,
                    help="MegaScenes root directory for random scene picking")
    p.add_argument("--output-dir", type=str, default="test_captions_out")
    p.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    p.add_argument("--model", type=str, default="OpenGVLab/InternVL3-38B")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.scenes:
        scene_dirs = [Path(s) for s in args.scenes]
    else:
        scene_dirs = _pick_random_scenes(args.megascenes_root, args.num_scenes)
        print(f"Randomly picked {len(scene_dirs)} scenes from {args.megascenes_root}\n")

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        images = discover_scene_images(scene_dir)
        if not images:
            print(f"SKIP {scene_name}: no images")
            continue

        targets = random.sample(images, min(args.num_images, len(images)))
        for target in targets:
            print(f"{scene_name}: picked {target.name}")

            try:
                caption = caption_single_image(target, args.api_base, args.model)
            except Exception as e:
                print(f"  FAIL: {e}")
                continue

            print(f"  caption: {caption}")

            img = Image.open(target).convert("RGB")
            labeled = draw_caption(img, caption)
            save_path = out / f"{scene_name}_{target.stem}.jpg"
            labeled.save(save_path, quality=90)
            print(f"  saved: {save_path}")

    print(f"\nDone. Results in {out}/")


if __name__ == "__main__":
    main()
