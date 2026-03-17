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


def main():
    p = argparse.ArgumentParser(description="Test VLM captioning on random scene targets")
    p.add_argument("--scenes", nargs="+", required=True, help="Scene directories")
    p.add_argument("--output-dir", type=str, default="test_captions_out")
    p.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for scene_path in args.scenes:
        scene_dir = Path(scene_path)
        scene_name = scene_dir.name
        images = discover_scene_images(scene_dir)
        if not images:
            print(f"SKIP {scene_name}: no images")
            continue

        target = random.choice(images)
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
