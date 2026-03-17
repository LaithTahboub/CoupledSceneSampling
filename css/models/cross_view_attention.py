"""Block-level multi-view self-attention inflation (CAT3D-style).

Instead of hooking into Diffusers' attention processor dispatch, we directly
replace the ``attn1`` (self-attention) module on each BasicTransformerBlock
with a ``MultiViewSelfAttention`` wrapper.  This is the "inflate 2D
self-attention into 3D" approach from CAT3D: tokens from all views are
concatenated along the sequence dimension before the QKV projection, so every
view attends to every other view.  Text cross-attention (attn2) is unchanged.

Usage::

    from css.models.cross_view_attention import inflate_unet_attention

    inflated = inflate_unet_attention(unet, num_views=3)
    print(f"Inflated {len(inflated)} self-attention layers")
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


# ---------------------------------------------------------------------------
# Multi-view self-attention module (replaces attn1 on selected blocks)
# ---------------------------------------------------------------------------

class MultiViewSelfAttention(nn.Module):
    """Wraps a Diffusers ``Attention`` module to do joint self-attention
    across all views.

    The UNet sees ``(B*V, S, C)`` tokens per block.  This module reshapes to
    ``(B, V*S, C)``, runs the original ``Attention`` (which holds the QKV
    weights and processor), then reshapes back.  Every spatial token therefore
    attends to tokens from every view — the core of the CAT3D 3D
    self-attention inflation.

    For batches whose size is not divisible by ``num_views`` (e.g. single-view
    inference), the module falls through to standard per-view attention.
    """

    def __init__(self, base_attn: nn.Module, num_views: int):
        super().__init__()
        if num_views < 2:
            raise ValueError(f"num_views must be >= 2, got {num_views}")
        self.base_attn = base_attn
        self.num_views = int(num_views)

    # Proxy attribute access so diffusers internals that inspect attn1
    # (e.g. ``block.attn1.to_q``, ``block.attn1.heads``) still work.
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_attn, name)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Cross-attention path (attn2 calls): pass through unchanged.
        if encoder_hidden_states is not None:
            return self.base_attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )

        v = self.num_views
        b, seq, c = hidden_states.shape

        # Fallback: batch not divisible by num_views → normal per-view attn.
        if b % v != 0:
            return self.base_attn(
                hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )

        bs = b // v

        # Pack: (B*V, S, C) → (B, V*S, C)
        packed = rearrange(
            hidden_states, "(bs v) s c -> bs (v s) c", bs=bs, v=v,
        )

        # Pack attention mask if present.
        packed_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                packed_mask = rearrange(
                    attention_mask, "(bs v) s -> bs (v s)", bs=bs, v=v,
                )
            elif attention_mask.ndim == 3:
                packed_mask = rearrange(
                    attention_mask, "(bs v) h s -> bs h (v s)", bs=bs, v=v,
                )

        # Run the original Attention on the packed sequence.
        packed_out = self.base_attn(
            packed,
            attention_mask=packed_mask,
            **kwargs,
        )

        # Unpack: (B, V*S, C) → (B*V, S, C)
        return rearrange(
            packed_out, "bs (v s) c -> (bs v) s c", v=v, s=seq,
        )


# ---------------------------------------------------------------------------
# UNet inflation: walk the module tree and replace attn1 in-place
# ---------------------------------------------------------------------------

# SD 2.1 @ 512: 64×64 feature maps live in down_blocks.0 and up_blocks.3.
# CAT3D: "we use 3D self-attention only in feature maps of size 32×32 and
# smaller" due to "significant computational overhead for relative small gain
# in fidelity."
_HIGH_RES_BLOCKS = {"down_blocks.0", "up_blocks.3"}


def _is_high_res_block(name: str) -> bool:
    for prefix in _HIGH_RES_BLOCKS:
        if name.startswith(prefix + "."):
            return True
    return False


def inflate_unet_attention(
    unet: nn.Module,
    num_views: int,
    include_high_res: bool = False,
) -> list[str]:
    """Inflate 2D self-attention layers in a UNet to multi-view 3D attention.

    Walks every ``BasicTransformerBlock`` (any module with both ``attn1`` and
    ``attn2`` attributes) and replaces ``attn1`` with a
    ``MultiViewSelfAttention`` wrapper.

    Args:
        unet: A ``UNet2DConditionModel`` (or compatible) to modify **in-place**.
        num_views: Number of views packed along the batch dimension.
        include_high_res: Also inflate 64×64 blocks (expensive, off by default).

    Returns:
        List of module paths that were inflated (for logging / verification).
    """
    inflated: list[str] = []

    for name, module in unet.named_modules():
        # BasicTransformerBlock has both attn1 (self) and attn2 (cross).
        if not (hasattr(module, "attn1") and hasattr(module, "attn2")):
            continue

        if not include_high_res and _is_high_res_block(name):
            continue

        attn1 = module.attn1

        # Don't double-wrap.
        if isinstance(attn1, MultiViewSelfAttention):
            continue

        module.attn1 = MultiViewSelfAttention(attn1, num_views)
        inflated.append(f"{name}.attn1")

    return inflated
