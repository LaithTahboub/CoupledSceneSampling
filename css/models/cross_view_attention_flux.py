"""Multi-view attention inflation for Flux transformer blocks.

Adapts the CAT3D-style multi-view self-attention from SD2.1 UNet to the Flux
DiT architecture.  Flux has two block types:

1. **Double-stream blocks** (`FluxTransformerBlock`):
   - `block.attn` is a `FluxAttention` with `added_kv_proj_dim` set (joint
     image+text attention).  The processor concatenates text and image Q/K/V,
     runs attention, then splits the output.
   - We wrap the *processor* to expand the image token portion across views
     before the QKV projection, then split back after attention.

2. **Single-stream blocks** (`FluxSingleTransformerBlock`):
   - `block.attn` is a `FluxAttention` with `pre_only=True`.  Text and image
     tokens are concatenated *before* entering the block, so the hidden_states
     already contain both.
   - We wrap the processor to expand the image portion across views.

In both cases, text tokens are *shared* across all views (not duplicated).
Only the image tokens are expanded so every spatial token can attend to tokens
from all views.

Usage::

    from css.models.cross_view_attention_flux import inflate_flux_attention

    inflated = inflate_flux_attention(transformer, num_views=3)
    print(f"Inflated {len(inflated)} attention layers")
"""

from __future__ import annotations

import torch
from einops import rearrange


# ---------------------------------------------------------------------------
# Multi-view processor for double-stream blocks (FluxTransformerBlock)
# ---------------------------------------------------------------------------

class MultiViewFluxAttnProcessor:
    """Wraps the standard FluxAttnProcessor to do joint attention across views.

    For double-stream blocks, `hidden_states` contains image tokens of shape
    (B*V, S_img, C) and `encoder_hidden_states` contains text tokens of shape
    (B*V, S_txt, C).  We reshape image tokens to (B, V*S_img, C), keep text
    tokens at (B, S_txt, C) (taking the first view's copy since they're
    identical), run the original processor, then reshape back.
    """

    def __init__(self, base_processor, num_views: int):
        self.base_processor = base_processor
        self.num_views = int(num_views)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
        **kwargs,
    ) -> torch.Tensor:
        v = self.num_views
        bv, s_img, c = hidden_states.shape

        # Fallback: batch not divisible by num_views → normal attention.
        if bv % v != 0:
            return self.base_processor(
                attn, hidden_states, encoder_hidden_states,
                attention_mask, image_rotary_emb, **kwargs,
            )

        bs = bv // v

        # Pack image tokens: (B*V, S_img, C) → (B, V*S_img, C)
        img_packed = rearrange(hidden_states, "(bs v) s c -> bs (v s) c", bs=bs, v=v)

        # Text tokens: all V copies are identical, take first view's
        txt_packed = None
        if encoder_hidden_states is not None:
            # (B*V, S_txt, C) → take every V-th → (B, S_txt, C)
            txt_packed = encoder_hidden_states[::v]

        # Expand RoPE embeddings for multi-view image tokens
        mv_rotary_emb = None
        if image_rotary_emb is not None:
            cos_emb, sin_emb = image_rotary_emb  # each (S_txt + S_img, head_dim)
            if encoder_hidden_states is not None:
                s_txt = encoder_hidden_states.shape[1]
            else:
                s_txt = 0
            # Split text and image portions of RoPE
            txt_cos, img_cos = cos_emb[:s_txt], cos_emb[s_txt:]
            txt_sin, img_sin = sin_emb[:s_txt], sin_emb[s_txt:]
            # Tile image RoPE across views
            img_cos_mv = img_cos.repeat(v, 1)
            img_sin_mv = img_sin.repeat(v, 1)
            # Recombine
            mv_rotary_emb = (
                torch.cat([txt_cos, img_cos_mv], dim=0),
                torch.cat([txt_sin, img_sin_mv], dim=0),
            )

        # Run base processor with packed multi-view tokens
        result = self.base_processor(
            attn, img_packed, txt_packed,
            attention_mask, mv_rotary_emb, **kwargs,
        )

        # Unpack results
        if isinstance(result, tuple) and len(result) == 2:
            # Double-stream: (img_out, txt_out)
            img_out, txt_out = result
            # (B, V*S_img, C) → (B*V, S_img, C)
            img_out = rearrange(img_out, "bs (v s) c -> (bs v) s c", v=v, s=s_img)
            # Expand text back to (B*V, S_txt, C)
            txt_out = txt_out.repeat_interleave(v, dim=0)
            return img_out, txt_out
        else:
            # Should not happen for double-stream, but handle gracefully
            out = rearrange(result, "bs (v s) c -> (bs v) s c", v=v, s=s_img)
            return out


# ---------------------------------------------------------------------------
# Multi-view processor for single-stream blocks (FluxSingleTransformerBlock)
# ---------------------------------------------------------------------------

class MultiViewFluxSingleAttnProcessor:
    """Wraps the FluxAttnProcessor for single-stream blocks.

    In single-stream blocks, the block's forward() concatenates text + image
    tokens before calling attn.  So hidden_states = [text; image] of shape
    (B*V, S_txt + S_img, C).

    We split text from image, pack image across views, concatenate with
    shared text, run attention, then split and unpack.
    """

    def __init__(self, base_processor, num_views: int, text_seq_len: int):
        self.base_processor = base_processor
        self.num_views = int(num_views)
        self.text_seq_len = int(text_seq_len)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
        **kwargs,
    ) -> torch.Tensor:
        v = self.num_views
        bv = hidden_states.shape[0]

        # Fallback
        if bv % v != 0:
            return self.base_processor(
                attn, hidden_states, encoder_hidden_states,
                attention_mask, image_rotary_emb, **kwargs,
            )

        bs = bv // v
        s_txt = self.text_seq_len
        s_img = hidden_states.shape[1] - s_txt

        # Split text and image portions
        txt_tokens = hidden_states[:, :s_txt]   # (B*V, S_txt, C)
        img_tokens = hidden_states[:, s_txt:]   # (B*V, S_img, C)

        # Pack image: (B*V, S_img, C) → (B, V*S_img, C)
        img_packed = rearrange(img_tokens, "(bs v) s c -> bs (v s) c", bs=bs, v=v)

        # Text: take first view's copy (B, S_txt, C)
        txt_shared = txt_tokens[::v]

        # Reconstruct combined: (B, S_txt + V*S_img, C)
        combined = torch.cat([txt_shared, img_packed], dim=1)

        # Expand RoPE for multi-view
        mv_rotary_emb = None
        if image_rotary_emb is not None:
            cos_emb, sin_emb = image_rotary_emb
            txt_cos, img_cos = cos_emb[:s_txt], cos_emb[s_txt:]
            txt_sin, img_sin = sin_emb[:s_txt], sin_emb[s_txt:]
            img_cos_mv = img_cos.repeat(v, 1)
            img_sin_mv = img_sin.repeat(v, 1)
            mv_rotary_emb = (
                torch.cat([txt_cos, img_cos_mv], dim=0),
                torch.cat([txt_sin, img_sin_mv], dim=0),
            )

        # Run base processor (single-stream: no encoder_hidden_states)
        out = self.base_processor(
            attn, combined, None,
            attention_mask, mv_rotary_emb, **kwargs,
        )

        # out shape: (B, S_txt + V*S_img, C)
        txt_out = out[:, :s_txt]
        img_out = out[:, s_txt:]

        # Unpack image: (B, V*S_img, C) → (B*V, S_img, C)
        img_out = rearrange(img_out, "bs (v s) c -> (bs v) s c", v=v, s=s_img)

        # Expand text back: (B, S_txt, C) → (B*V, S_txt, C)
        txt_out = txt_out.repeat_interleave(v, dim=0)

        # Recombine for the block's post-processing
        return torch.cat([txt_out, img_out], dim=1)


# ---------------------------------------------------------------------------
# Inflate function: walk the transformer and replace processors
# ---------------------------------------------------------------------------

def inflate_flux_attention(
    transformer,
    num_views: int,
    text_seq_len: int = 512,
) -> list[str]:
    """Inflate attention in a FluxTransformer2DModel to multi-view.

    Args:
        transformer: A ``FluxTransformer2DModel`` to modify **in-place**.
        num_views: Number of views packed along the batch dimension.
        text_seq_len: Maximum text sequence length (T5 tokens, default 512).

    Returns:
        List of module paths that were inflated.
    """
    inflated: list[str] = []

    # Double-stream blocks
    for i, block in enumerate(transformer.transformer_blocks):
        base_proc = block.attn.processor
        if isinstance(base_proc, MultiViewFluxAttnProcessor):
            continue  # already inflated
        block.attn.set_processor(
            MultiViewFluxAttnProcessor(base_proc, num_views)
        )
        inflated.append(f"transformer_blocks.{i}.attn")

    # Single-stream blocks
    for i, block in enumerate(transformer.single_transformer_blocks):
        base_proc = block.attn.processor
        if isinstance(base_proc, MultiViewFluxSingleAttnProcessor):
            continue
        block.attn.set_processor(
            MultiViewFluxSingleAttnProcessor(base_proc, num_views, text_seq_len)
        )
        inflated.append(f"single_transformer_blocks.{i}.attn")

    return inflated
