"""Batch VLM captioning for MegaScenes target images.
Usage:
    # 1. Start vLLM server (on one or more GPUs):
    vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
        --host 0.0.0.0 --port 8000 \
        --tensor-parallel-size 2 \
        --max-model-len 4096 \
        --limit-mm-per-prompt image=1

    # 2. Run captioning:
    python -m css.data.caption_dataset \
        --scenes-file scenes.txt \
        --output-dir captions/ \
        --api-base http://localhost:8000/v1 \
        --batch-size 64 \
        --num-workers 8

    # 3. (Optional) Enforce consistent scene_type within each scene:
    python -m css.data.caption_dataset \
        --output-dir captions/ \
        --scene-type-override
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image
import openai


SYSTEM_PROMPT = """
You are an image captioning system for a scene relighting dataset. Given a photograph, produce a single concise caption describing the scene's lighting, atmosphere, and transient properties.

The caption should read as one natural descriptive sentence in the following flow:

"A [scene type], [lighting and shadow description], [sky and weather], [time of day and season], [transient objects if any]."

Focus on:
- Scene type as a brief noun phrase (e.g., "a stone bridge over a river", "a cobblestone plaza"). Do NOT name specific landmarks.
- Lighting: source type, direction, color/temperature (e.g., "warm amber", "cool bluish-white"), shadow hardness and direction, overall contrast. Mention fill or secondary light only if clearly visible.
- Sky and weather: sky color, cloud type, atmospheric conditions like fog, haze, rain, or wet ground reflections.
- Time and season: state if inferable from shadows, light angle, foliage, snow, or clothing.
- Transient/temporary objects: people, vehicles, animals, scaffolding, etc., with approximate position in frame. Omit if none.

Use specific color terms ("warm golden-amber light", "pale gray overcast sky", "cool steel-blue shadows") rather than vague descriptions. Only describe what is visible. Output ONLY the caption — no explanation, no labels, no line breaks.
"""
USER_PROMPT = "Describe this photograph."

_thread_local = threading.local()


def get_client(api_base: str) -> openai.OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = openai.OpenAI(base_url=api_base, api_key="dummy")
        _thread_local.client = client
    return client


def encode_image_base64(image_path: Path, max_size: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def caption_single_image(
    image_path: Path,
    api_base: str,
    model: str,
    temperature: float = 0.23,
    max_tokens: int = 100,
    max_image_size: int = 1024,
) -> str:
    client = get_client(api_base)
    b64 = encode_image_base64(image_path, max_size=max_image_size)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}",
                    },
                ],
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip().strip('"')


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def discover_scene_images(scene_dir: Path) -> list[Path]:
    """Return all image files under scene_dir/images/."""
    images_dir = scene_dir / "images"
    if not images_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def load_existing_captions(caption_file: Path) -> dict[str, str]:
    """Load existing captions JSON if it exists."""
    if caption_file.exists():
        with open(caption_file) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Batch captioning for one scene
# ---------------------------------------------------------------------------

def caption_scene(
    scene_dir: Path,
    output_dir: Path,
    api_base: str,
    model: str,
    *,
    batch_size: int = 128,
    num_workers: int = 12,
    temperature: float = 0.23,
    max_tokens: int = 100,
    max_image_size: int = 1024,
    resume: bool = True,
) -> dict[str, str]:
    scene_name = scene_dir.name
    caption_file = output_dir / f"{scene_name}.json"
    captions = load_existing_captions(caption_file) if resume else {}

    images = discover_scene_images(scene_dir)
    if not images:
        print(f"  SKIP {scene_name}: no images found")
        return captions

    images_dir = scene_dir / "images"
    todo = []
    for img_path in images:
        rel = img_path.relative_to(images_dir).as_posix()
        if rel not in captions:
            todo.append((rel, img_path))

    if not todo:
        print(f"  {scene_name}: all {len(captions)} images already captioned")
        return captions

    print(f"  {scene_name}: {len(todo)} images to caption ({len(captions)} already done)")

    failed = 0
    completed_since_save = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for rel, img_path in todo:
            fut = executor.submit(
                caption_single_image,
                img_path, api_base, model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_image_size=max_image_size,
            )
            futures[fut] = rel

        for fut in as_completed(futures):
            rel = futures[fut]
            try:
                captions[rel] = fut.result()
            except Exception as e:
                print(f"    FAIL {rel}: {e}")
                failed += 1

            completed_since_save += 1
            if completed_since_save >= batch_size:
                with open(caption_file, "w") as f:
                    json.dump(captions, f)
                completed_since_save = 0

    with open(caption_file, "w") as f:
        json.dump(captions, f)

    n_new = len(todo) - failed
    print(f"  {scene_name}: captioned {n_new} new images ({failed} failed)")
    return captions


# ---------------------------------------------------------------------------
# Scene-type override (post-processing)
# ---------------------------------------------------------------------------

_SCENE_TYPE_RE = re.compile(r"^A photo of (.+?),\s")


def extract_scene_type(caption: str) -> str | None:
    """Extract the [scene_type] from a structured caption."""
    m = _SCENE_TYPE_RE.match(caption)
    return m.group(1) if m else None


def apply_scene_type_override(caption_file: Path) -> None:
    """Replace each caption's scene_type with the most common one in the scene."""
    if not caption_file.exists():
        return

    with open(caption_file) as f:
        captions: dict[str, str] = json.load(f)

    if not captions:
        return

    # Count scene types
    types = []
    for cap in captions.values():
        st = extract_scene_type(cap)
        if st:
            types.append(st)

    if not types:
        return

    most_common = Counter(types).most_common(1)[0][0]

    # Replace
    updated = 0
    for key, cap in captions.items():
        st = extract_scene_type(cap)
        if st and st != most_common:
            captions[key] = cap.replace(f"A photo of {st},", f"A photo of {most_common},", 1)
            updated += 1

    if updated:
        with open(caption_file, "w") as f:
            json.dump(captions, f, indent=2)
        print(f"  {caption_file.stem}: unified scene_type to '{most_common}' ({updated} updated)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch VLM captioning for MegaScenes")

    p.add_argument("--scenes", nargs="*", default=None,
                    help="Scene directories to caption")
    p.add_argument("--scenes-file", type=str, default=None,
                    help="Text file with one scene directory per line")
    p.add_argument("--output-dir", type=str, required=True,
                    help="Directory to save per-scene caption JSON files")
    p.add_argument("--api-base", type=str, default="http://localhost:8000/v1",
                    help="vLLM OpenAI-compatible API base URL")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                    help="Model name as registered in vLLM")
    p.add_argument("--batch-size", type=int, default=128,
                    help="Number of images to process per save-batch")
    p.add_argument("--num-workers", type=int, default=12,
                    help="Concurrent API requests per batch")
    p.add_argument("--temperature", type=float, default=0.23)
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--max-image-size", type=int, default=1024,
                    help="Resize images so max dimension <= this value")
    p.add_argument("--no-resume", action="store_true",
                    help="Do not skip already-captioned images")
    p.add_argument("--scene-type-override", action="store_true",
                    help="Post-process: unify scene_type within each scene")

    return p.parse_args()


def _read_lines(path: str | None) -> list[str]:
    if path is None:
        return []
    lines = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = list(dict.fromkeys((args.scenes or []) + _read_lines(args.scenes_file)))

    if args.scene_type_override and not scenes:
        # Apply override to all existing caption files
        print("Applying scene_type override to all caption files...")
        for caption_file in sorted(output_dir.glob("*.json")):
            apply_scene_type_override(caption_file)
        return

    if not scenes:
        print("No scenes specified. Use --scenes or --scenes-file.", file=sys.stderr)
        sys.exit(1)

    print(f"Captioning {len(scenes)} scenes -> {output_dir}")

    for scene_path in scenes:
        scene_dir = Path(scene_path)
        if not scene_dir.is_dir():
            print(f"  SKIP {scene_path}: not a directory")
            continue

        caption_scene(
            scene_dir, output_dir, args.api_base, args.model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_image_size=args.max_image_size,
            resume=not args.no_resume,
        )

    if args.scene_type_override:
        print("\nApplying scene_type override...")
        for scene_path in scenes:
            scene_name = Path(scene_path).name
            apply_scene_type_override(output_dir / f"{scene_name}.json")

    print("Done.")


if __name__ == "__main__":
    main()
