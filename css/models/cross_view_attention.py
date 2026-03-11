from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from einops import rearrange


class CrossViewAttentionProcessor(nn.Module):
    """
    self-attn processor that mixes tokens across views.
    """

    def __init__(self, num_views: int, base_processor: Any):
        super().__init__()
        if num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {num_views}")
        self.num_views = int(num_views)
        self.base_processor = base_processor

    def _call_base(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        temb: torch.Tensor | None,
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

    def _pack_attention_mask(
        self,
        attention_mask: torch.Tensor,
        *,
        bs: int,
        v: int,
        seq: int,
    ) -> torch.Tensor | None:
        if attention_mask.ndim == 2:
            b, s = attention_mask.shape
            if s != seq:
                return None
            # (bs*v, seq) -> (bs, v*seq)
            return rearrange(attention_mask, "(bs v) s -> bs (v s)", bs=bs, v=v, s=seq)

        if attention_mask.ndim == 3:
            b, one, s = attention_mask.shape
            if one != 1 or s != seq:
                return None
            # (bs*v, 1, seq) -> (bs, 1, v*seq)
            return rearrange(attention_mask, "(bs v) 1 s -> bs 1 (v s)", bs=bs, v=v, s=seq)

        return None

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
        # dont change text conditioning
        if encoder_hidden_states is not None:
            return self._call_base(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

        # we support token inputs (b, s, c) and spatial inputs (b, c, h, w)
        if hidden_states.ndim not in (3, 4):
            return self._call_base(
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
            tokens = rearrange(hidden_states, "b c h w -> b (h w) c")
        else:
            b, seq, c = hidden_states.shape
            tokens = hidden_states
            h = w = None

        v = self.num_views
        if b % v != 0:
            return self._call_base(
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

        bs = b // v

        # (bs*v, seq, c) -> (bs, v*seq, c)
        packed_tokens = rearrange(tokens, "(bs v) s c -> bs (v s) c", bs=bs, v=v, s=seq)

        # if a mask is present, try to pack it
        packed_mask = None
        if attention_mask is not None:
            packed_mask = self._pack_attention_mask(attention_mask, bs=bs, v=v, seq=seq)
            if packed_mask is None:
                return self._call_base(
                    attn,
                    hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=attention_mask,
                    temb=temb,
                    *args,
                    **kwargs,
                )

        # run base processor once on packed tokens (this is where views mix)
        packed_out = self._call_base(
            attn,
            packed_tokens,
            encoder_hidden_states=None,
            attention_mask=packed_mask,
            temb=temb,  
            *args,
            **kwargs,
        )

        # unpack: (bs, v*seq, c) -> (bs*v, seq, c)
        out_tokens = rearrange(packed_out, "bs (v s) c -> (bs v) s c", v=v, s=seq)

        if is_4d:
            out = rearrange(out_tokens, "b (h w) c -> b c h w", h=h, w=w)
        else:
            out = out_tokens

        return out


class CrossViewAttention:
    """attach cross-view self-attn processors to a diffusers unet."""

    def __init__(self, num_views: int, include_high_res: bool = False):
        if num_views < 2:
            raise ValueError("cross-view attention expects at least 2 views")
        self.num_views = int(num_views)
        self.include_high_res = bool(include_high_res)
        self.enabled_processor_names: list[str] = []

    def _should_wrap(self, name: str) -> bool:
        # diffusers convention is attn1 = self-attn, attn2 = cross-attn
        if not name.endswith("attn1.processor"):
            return False
        if self.include_high_res:
            return True

        # sd2.1 @ 512: 64x64 self-attn lives in down_blocks.0 and up_blocks.3
        # cat3d skips 64x64 by default due to cost,we also do that
        if name.startswith("down_blocks.0.") or name.startswith("up_blocks.3."):
            return False
        return True

    def attach(self, unet: Any) -> None:
        processors: dict[str, Any] = {}
        self.enabled_processor_names = []

        for name, proc in unet.attn_processors.items():
            base = proc.base_processor if isinstance(proc, CrossViewAttentionProcessor) else proc
            if self._should_wrap(name):
                processors[name] = CrossViewAttentionProcessor(self.num_views, base)
                self.enabled_processor_names.append(name)
            else:
                processors[name] = base

        unet.set_attn_processor(processors)

    def set_num_views(self, n: int, unet: Any) -> None:
        """Dynamically update the view count on all wrapped processors.

        Needed for multi-target training where the number of views per
        example varies between steps.  Handles DDP-wrapped unets.
        """
        self.num_views = n
        module = unet.module if hasattr(unet, "module") else unet
        for name, proc in module.attn_processors.items():
            if isinstance(proc, CrossViewAttentionProcessor):
                proc.num_views = n

    @property
    def num_wrapped(self) -> int:
        return len(self.enabled_processor_names)