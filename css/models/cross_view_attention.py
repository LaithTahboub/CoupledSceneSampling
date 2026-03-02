from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class CrossViewAttentionProcessor(nn.Module):
    """Self-attention processor that mixes tokens across views in the batch.

    Expected input layout is `(batch * num_views, ...)` where views are packed
    in fixed order per scene. For self-attention layers, tokens from all views
    of the same scene are concatenated along sequence length so attention can
    exchange information across views. Cross-attention layers are untouched.
    """

    def __init__(self, num_views: int, base_processor: Any):
        super().__init__()
        if num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {num_views}")
        self.num_views = int(num_views)
        self.base_processor = base_processor

    def _fallback(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )

    def forward(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Keep cross-attention path exactly as original.
        if encoder_hidden_states is not None:
            return self._fallback(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

        # If attention carries extra masks/embeddings, avoid shape assumptions.
        if attention_mask is not None or temb is not None:
            return self._fallback(
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

        if hidden_states.ndim not in (3, 4):
            return self._fallback(
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

        is_4d = hidden_states.ndim == 4
        if is_4d:
            b, c, h, w = hidden_states.shape
            seq = h * w
            tokens = hidden_states.permute(0, 2, 3, 1).reshape(b, seq, c)
        else:
            b, seq, c = hidden_states.shape
            tokens = hidden_states
            h = w = 0

        if b % self.num_views != 0:
            return self._fallback(
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

        batch_scenes = b // self.num_views
        packed = tokens.reshape(batch_scenes, self.num_views * seq, c)

        mixed = self._fallback(
            attn,
            packed,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            *args,
            **kwargs,
        )

        mixed = mixed.reshape(batch_scenes, self.num_views, seq, c).reshape(b, seq, c)
        if is_4d:
            mixed = mixed.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return mixed


class CrossViewAttention:
    """Attach CAT3D-style cross-view attention processors to a UNet."""

    def __init__(self, num_views: int, include_high_res: bool = False):
        if num_views < 2:
            raise ValueError("CrossViewAttention expects at least 2 views")
        self.num_views = int(num_views)
        self.include_high_res = bool(include_high_res)
        self.enabled_processor_names: list[str] = []

    def _should_wrap(self, name: str) -> bool:
        # `attn1` is self-attention in diffusers' transformer blocks.
        if not name.endswith("attn1.processor"):
            return False
        if self.include_high_res:
            return True

        # SD2.1 @ 512 has 64x64 attention at down_blocks.0 and up_blocks.3.
        # CAT3D applies cross-view attention at 32x32 and below.
        if name.startswith("down_blocks.0.") or name.startswith("up_blocks.3."):
            return False
        return True

    def attach(self, unet: Any) -> None:
        processors = {}
        self.enabled_processor_names = []

        for name, processor in unet.attn_processors.items():
            base = processor.base_processor if isinstance(processor, CrossViewAttentionProcessor) else processor
            if self._should_wrap(name):
                processors[name] = CrossViewAttentionProcessor(self.num_views, base)
                self.enabled_processor_names.append(name)
            else:
                processors[name] = base

        unet.set_attn_processor(processors)

    @property
    def num_wrapped(self) -> int:
        return len(self.enabled_processor_names)
