from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class MultiViewSelfAttnProcessor(nn.Module):
    """Explicit multi-view self-attention for diffusers UNet `attn1`.

    Behavior:
    - Queries come from each view independently.
    - Keys/values come from token concatenation over all views.

    Expected self-attn token layout is `(B * V, L, D)` where views are packed
    in fixed order for each scene.
    """

    def __init__(self, num_views: int, base_processor: Any):
        super().__init__()
        if num_views < 2:
            raise ValueError(f"num_views must be >= 2, got {num_views}")
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

    def _apply_optional_qk_norm(
        self,
        attn,
        q: torch.Tensor,  # (B, V, L, D)
        k: torch.Tensor,  # (B, V*L, D)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_q = getattr(attn, "norm_q", None)
        norm_k = getattr(attn, "norm_k", None)

        if norm_q is not None:
            b, v, l, d = q.shape
            q = norm_q(q.reshape(b * v, l, d)).reshape(b, v, l, d)
        if norm_k is not None:
            b, vl, d = k.shape
            k = norm_k(k.reshape(b, vl, d)).reshape(b, vl, d)
        return q, k

    def _scaled_dot_attn(
        self,
        attn,
        q: torch.Tensor,  # (B, V, H, L, Dh)
        k: torch.Tensor,  # (B, H, V*L, Dh)
        v: torch.Tensor,  # (B, H, V*L, Dh)
    ) -> torch.Tensor:
        head_dim = q.shape[-1]
        scale = getattr(attn, "scale", None)
        if scale is None:
            scale = head_dim**-0.5

        q_scores = q
        k_scores = k
        if getattr(attn, "upcast_attention", False):
            q_scores = q_scores.float()
            k_scores = k_scores.float()

        scores = torch.matmul(q_scores, k_scores.transpose(-1, -2)) * scale  # (B, V, H, L, V*L)
        if getattr(attn, "upcast_softmax", False):
            probs = scores.float().softmax(dim=-1).to(dtype=q.dtype)
        else:
            probs = scores.softmax(dim=-1)

        return torch.matmul(probs.to(dtype=v.dtype), v)  # (B, V, H, L, Dh)

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
        # This processor is for self-attn only; leave cross-attn untouched.
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

        # Keep unsupported/rare paths delegated to baseline behavior.
        if attention_mask is not None:
            return self._fallback(
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

        residual = hidden_states

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            bv, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(bv, c, h * w).transpose(1, 2)  # (BV, L, D)
            l_tokens = h * w
        elif input_ndim == 3:
            bv, l_tokens, c = hidden_states.shape
            h = w = 0
        else:
            return self._fallback(
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        v_count = self.num_views
        if bv % v_count != 0:
            raise ValueError(f"Expected BV divisible by V, got BV={bv}, V={v_count}")
        b = bv // v_count
        d_model = hidden_states.shape[-1]

        # (B, V, L, D)
        x = hidden_states.reshape(b, v_count, l_tokens, d_model)

        # Queries from each view, K/V from concatenated tokens across all views.
        q = attn.to_q(x)                           # (B, V, L, D)
        x_kv = x.reshape(b, v_count * l_tokens, d_model)
        k = attn.to_k(x_kv)                        # (B, V*L, D)
        v = attn.to_v(x_kv)                        # (B, V*L, D)
        q, k = self._apply_optional_qk_norm(attn, q, k)

        heads = int(attn.heads)
        if d_model % heads != 0:
            raise ValueError(f"D={d_model} not divisible by heads={heads}")
        d_head = d_model // heads

        q = q.view(b, v_count, l_tokens, heads, d_head).permute(0, 1, 3, 2, 4)  # (B, V, H, L, Dh)
        k = k.view(b, v_count * l_tokens, heads, d_head).permute(0, 2, 1, 3)     # (B, H, V*L, Dh)
        v = v.view(b, v_count * l_tokens, heads, d_head).permute(0, 2, 1, 3)     # (B, H, V*L, Dh)

        out = self._scaled_dot_attn(attn, q, k, v)                                 # (B, V, H, L, Dh)
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(b, v_count, l_tokens, d_model)
        out = out.view(bv, l_tokens, d_model)

        # diffusers stores output as [linear_proj, dropout]
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if input_ndim == 4:
            out = out.transpose(1, 2).reshape(bv, c, h, w)

        if getattr(attn, "residual_connection", False):
            out = out + residual
        out = out / getattr(attn, "rescale_output_factor", 1.0)
        return out


class CrossViewAttention:
    """Attach explicit multi-view self-attn processors to a UNet."""

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
            base = processor.base_processor if isinstance(processor, MultiViewSelfAttnProcessor) else processor
            if self._should_wrap(name):
                processors[name] = MultiViewSelfAttnProcessor(self.num_views, base)
                self.enabled_processor_names.append(name)
            else:
                processors[name] = base

        unet.set_attn_processor(processors)

    @property
    def num_wrapped(self) -> int:
        return len(self.enabled_processor_names)
