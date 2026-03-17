"""Inference-time image captioning utility.

Captions a single image using the same VLM and structured prompt used for
dataset captioning, producing a caption suitable for PoseSD text conditioning.

Usage as CLI:
    # Start vLLM server first, then:
    python -m css.inference.caption_image path/to/photo.jpg

Usage as library:
    from css.inference.caption_image import ImageCaptioner

    captioner = ImageCaptioner(api_base="http://localhost:8000/v1")
    caption = captioner("path/to/photo.jpg")
    # -> "A photo of a Gothic cathedral, under soft overcast light, ..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

from css.data.caption_dataset import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    encode_image_base64,
)


class ImageCaptioner:
    """Caption a single image using a VLM for PoseSD text conditioning.

    Can be used as a callable: captioner(image_path) -> caption string.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        temperature: float = 0.3,
        max_tokens: int = 120,
        max_image_size: int = 1024,
    ):
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_image_size = max_image_size
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(base_url=self.api_base, api_key="dummy")
        return self._client

    def __call__(self, image_path: str | Path) -> str:
        """Caption an image, returning a structured description string."""
        client = self._get_client()
        b64 = encode_image_base64(Path(image_path), max_size=self.max_image_size)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip().strip('"')


def main() -> None:
    p = argparse.ArgumentParser(description="Caption an image for PoseSD")
    p.add_argument("image", type=str, help="Path to the image")
    p.add_argument("--api-base", type=str, default="http://localhost:8000/v1")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=120)
    args = p.parse_args()

    captioner = ImageCaptioner(
        api_base=args.api_base,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    caption = captioner(args.image)
    print(caption)


if __name__ == "__main__":
    main()
