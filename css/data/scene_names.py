"""Utilities for robust scene identifiers and human-readable scene text."""

import re
from pathlib import Path


def derive_scene_key(scene_dir: str | Path) -> str:
    p = Path(scene_dir)
    parts = p.parts
    if "MegaScenes" in parts:
        idx = max(i for i, part in enumerate(parts) if part == "MegaScenes")
        tail = parts[idx + 1 :]
        if tail:
            return "/".join(tail)
    return p.name


def scene_pair_key(scene_key: str, image_name: str) -> str:
    return f"{scene_key}\t{image_name}"


def scene_prompt_name(scene_key: str) -> str:
    text = scene_key.replace("_", " ").replace("/", " ")
    text = text.replace('"', "")
    text = re.sub(r"(^|\s)'([^']+)'(\s|$)", r"\1\2\3", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
