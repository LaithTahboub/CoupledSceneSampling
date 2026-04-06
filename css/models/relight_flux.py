"""RelightFlux: 3-view model with Plucker ray conditioning on Flux.1-dev.

Analogous to RelightSD but built on Flux's DiT architecture.

Views are packed per-view as (B, 23, h, w) with channels:
    latent(16) + plucker(6) + mask(1)
then 2x2-packed into tokens of dim 92 before the transformer.

mask=1 for reference views (clean), mask=0 for target views (noised).

Supports:
- 3-view mode: [target, ref1, ref2]
- Randomized slot order during training
- Variable conditioning count (1 or 2 refs)
- Structured conditioning dropout (independent ref1/ref2/text dropping)
- Flow matching training (velocity prediction)
- True CFG at inference (2+ forward passes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from css.models.cross_view_attention_flux import inflate_flux_attention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_x_embedder(transformer: FluxTransformer2DModel, new_in_channels: int) -> None:
    """Expand the x_embedder linear layer to accept extra input channels.

    Analogous to _expand_conv_in for SD2.1 UNet.  The extra channels (plucker +
    mask) are zero-initialised so the model starts from pretrained behavior.
    """
    old_linear = transformer.x_embedder
    old_in = old_linear.in_features
    if old_in == new_in_channels:
        return

    new_linear = nn.Linear(
        new_in_channels, old_linear.out_features,
        bias=(old_linear.bias is not None),
    ).to(device=old_linear.weight.device, dtype=old_linear.weight.dtype)

    with torch.no_grad():
        new_linear.weight.zero_()
        new_linear.weight[:, :old_in].copy_(old_linear.weight)
        if old_linear.bias is not None:
            new_linear.bias.copy_(old_linear.bias)

    transformer.x_embedder = new_linear
    transformer.config.in_channels = new_in_channels


def _pack_latents(latents: torch.Tensor, num_channels: int) -> torch.Tensor:
    """Pack (B, C, H, W) into (B, S, C*4) via 2x2 patches.

    H and W must be divisible by 2.
    """
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
    return latents


def _unpack_latents(latents: torch.Tensor, h: int, w: int, channels: int) -> torch.Tensor:
    """Unpack (B, S, C*4) back to (B, C, H, W)."""
    b, s, _ = latents.shape
    latents = latents.view(b, h // 2, w // 2, channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(b, channels, h, w)
    return latents


def _prepare_latent_image_ids(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create positional IDs for latent image tokens (after 2x2 packing).

    Args:
        h, w: Spatial dims of the packed grid (= latent_h // 2, latent_w // 2).
    """
    ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
    ids[..., 1] = ids[..., 1] + torch.arange(h, device=device, dtype=dtype)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w, device=device, dtype=dtype)[None, :]
    return ids.reshape(h * w, 3)


def _prepare_text_ids(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Text IDs are all zeros (no spatial position for text tokens)."""
    return torch.zeros(seq_len, 3, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# RelightFlux model
# ---------------------------------------------------------------------------

class RelightFlux(nn.Module):
    """3-view model: 2 refs + 1 target with Plucker conditioning on Flux.1-dev."""

    NUM_VIEWS = 3
    LATENT_CH = 16         # Flux VAE latent channels
    PER_VIEW_CH = 23       # latent(16) + plucker(6) + mask(1)
    PACKED_TOKEN_DIM = 92  # PER_VIEW_CH * 4 (2x2 packing)
    VAE_SCALE_FACTOR = 8
    T5_MAX_LEN = 512

    def __init__(
        self,
        pretrained_model: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda",
        text_max_length: int = 512,
    ):
        super().__init__()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.text_max_length = text_max_length

        # --- Load Flux components ---
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model, subfolder="vae",
        ).to(self.device)

        self.transformer = FluxTransformer2DModel.from_pretrained(
            pretrained_model, subfolder="transformer",
        ).to(self.device)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model, subfolder="scheduler",
        )

        # CLIP text encoder (pooled output only)
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model, subfolder="text_encoder",
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model, subfolder="tokenizer",
        )

        # T5 text encoder (sequence output)
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            pretrained_model, subfolder="text_encoder_2",
        )
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            pretrained_model, subfolder="tokenizer_2",
        )

        # Freeze encoders and VAE
        self.vae.requires_grad_(False).eval()
        self.text_encoder.requires_grad_(False).eval()
        self.text_encoder_2.requires_grad_(False).eval()
        self.text_device = torch.device("cpu")

        # --- Expand input projection for per-view channels ---
        _expand_x_embedder(self.transformer, self.PACKED_TOKEN_DIM)

        # --- Inflate attention for multi-view ---
        inflated = inflate_flux_attention(
            self.transformer,
            num_views=self.NUM_VIEWS,
            text_seq_len=self.text_max_length,
        )
        print(f"[relight-flux] inflated {len(inflated)} attention layers to multi-view")

        # --- Precompute null text embeddings ---
        with torch.no_grad():
            null_clip_emb, null_t5_emb = self._encode_text_raw([""])
            self.register_buffer("null_clip_pooled", null_clip_emb, persistent=False)
            self.register_buffer("null_t5_emb", null_t5_emb, persistent=False)

    # ------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------

    def configure_trainable(self, train_mode: str = "full") -> None:
        if train_mode == "full":
            self.transformer.requires_grad_(True)
            return
        # "cond" mode: only x_embedder + attention layers
        self.transformer.requires_grad_(False)
        self.transformer.x_embedder.requires_grad_(True)
        for name, param in self.transformer.named_parameters():
            if "attn" in name:
                param.requires_grad_(True)

    def configure_memory_optimizations(
        self,
        gradient_checkpointing: bool = False,
        compile_transformer: bool = False,
    ) -> None:
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        if compile_transformer:
            mode = "default" if gradient_checkpointing else "reduce-overhead"
            self.transformer = torch.compile(self.transformer, mode=mode)
            print(f"[relight-flux] compiled transformer with torch.compile(mode='{mode}')")

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def _conditioning_device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    def _conditioning_dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_text_raw(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts with both CLIP (pooled) and T5 (sequence).

        The Flux text encoders are frozen and permanently kept on CPU to avoid
        repeatedly materializing T5-XXL on GPU during training. Only the much
        smaller embedding tensors are transferred to the transformer device.

        Returns:
            clip_pooled: (B, 768) - CLIP pooled embedding on the transformer device
            t5_hidden:   (B, S, 4096) - T5 sequence embedding on the transformer device
        """
        out_device = self._conditioning_device()
        out_dtype = self._conditioning_dtype()

        clip_tokens = self.tokenizer(
            prompts, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        )
        clip_out = self.text_encoder(
            clip_tokens.input_ids.to(self.text_device),
            attention_mask=clip_tokens.attention_mask.to(self.text_device),
        )
        clip_pooled = clip_out.pooler_output.to(device=out_device, dtype=out_dtype)

        t5_tokens = self.tokenizer_2(
            prompts, padding="max_length", max_length=self.text_max_length,
            truncation=True, return_tensors="pt",
        )
        t5_out = self.text_encoder_2(
            t5_tokens.input_ids.to(self.text_device),
            attention_mask=t5_tokens.attention_mask.to(self.text_device),
        )
        t5_hidden = t5_out.last_hidden_state.to(device=out_device, dtype=out_dtype)

        return clip_pooled, t5_hidden

    @torch.no_grad()
    def get_text_embeddings(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Public text encoding interface."""
        return self._encode_text_raw(prompts)

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """Encode image to latent. Applies Flux VAE scaling + shift."""
        latent_dist = self.vae.encode(img).latent_dist
        latent = latent_dist.sample()
        # Flux uses: encoded = (sample - shift) * scale  (inverted at decode)
        latent = (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image. Inverts Flux VAE scaling + shift."""
        latent = (latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(latent, return_dict=False)[0]

    # ------------------------------------------------------------------
    # View packing
    # ------------------------------------------------------------------

    def _pack_views(
        self,
        ref1_lat: torch.Tensor, ref2_lat: torch.Tensor, tgt_lat: torch.Tensor,
        pl_ref1: torch.Tensor, pl_ref2: torch.Tensor, pl_tgt: torch.Tensor,
        ref1_keep: torch.Tensor | None = None,
        ref2_keep: torch.Tensor | None = None,
        slot_order: list[int] | None = None,
        keep_pluckers: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Pack 3 views into (B*3, S, token_dim) for the transformer.

        Each view: latent(16) + plucker(6) + mask(1) = 23 channels at (h, w),
        then 2x2-packed into tokens of dim 92.

        Returns:
            packed: (B*3, S, 92) packed multi-view tokens
            target_slot: which slot index contains the target view
        """
        b, _, h, w = tgt_lat.shape
        dtype, device = tgt_lat.dtype, tgt_lat.device

        m_ref1 = torch.ones((b, 1, h, w), device=device, dtype=dtype)
        m_ref2 = torch.ones((b, 1, h, w), device=device, dtype=dtype)
        m_tgt = torch.zeros((b, 1, h, w), device=device, dtype=dtype)

        if ref1_keep is not None:
            keep1 = ref1_keep.to(device=device, dtype=dtype).view(b, 1, 1, 1)
            ref1_lat = ref1_lat * keep1
            if not keep_pluckers:
                pl_ref1 = pl_ref1 * keep1
            m_ref1 = m_ref1 * keep1

        if ref2_keep is not None:
            keep2 = ref2_keep.to(device=device, dtype=dtype).view(b, 1, 1, 1)
            ref2_lat = ref2_lat * keep2
            if not keep_pluckers:
                pl_ref2 = pl_ref2 * keep2
            m_ref2 = m_ref2 * keep2

        # Concatenate per-view channels: (B, 23, h, w)
        v_tgt = torch.cat([tgt_lat, pl_tgt, m_tgt], dim=1)
        v_r1 = torch.cat([ref1_lat, pl_ref1, m_ref1], dim=1)
        v_r2 = torch.cat([ref2_lat, pl_ref2, m_ref2], dim=1)

        views = [v_tgt, v_r1, v_r2]

        if slot_order is not None:
            ordered = [views[i] for i in slot_order]
            target_slot = slot_order.index(0)
        else:
            ordered = views
            target_slot = 0

        # Stack and reshape to (B*3, 23, h, w)
        stacked = torch.stack(ordered, dim=1).reshape(
            b * self.NUM_VIEWS, self.PER_VIEW_CH, h, w,
        )

        # 2x2 pack into tokens: (B*3, S, 92)
        packed = _pack_latents(stacked, self.PER_VIEW_CH)

        return packed, target_slot

    def _predict_target_velocity(
        self,
        packed: torch.Tensor,
        timesteps: torch.Tensor,
        clip_pooled: torch.Tensor,
        t5_hidden: torch.Tensor,
        batch_size: int,
        target_slot: int = 0,
        guidance: torch.Tensor | None = None,
        latent_h: int = 128,
        latent_w: int = 128,
    ) -> torch.Tensor:
        """Run the Flux transformer on packed multi-view tokens.

        Returns the predicted velocity for the target slot only.
        """
        V = self.NUM_VIEWS

        # Repeat timesteps for all views
        t = timesteps.repeat_interleave(V)

        # Repeat text embeddings for all views (will be collapsed in attn processor)
        pooled = clip_pooled.repeat_interleave(V, dim=0)
        t5 = t5_hidden.repeat_interleave(V, dim=0)

        # Prepare positional IDs
        packed_h, packed_w = latent_h // 2, latent_w // 2
        img_ids = _prepare_latent_image_ids(packed_h, packed_w, packed.device, packed.dtype)
        txt_ids = _prepare_text_ids(t5_hidden.shape[1], packed.device, packed.dtype)

        # Guidance embedding (required for Flux dev which has guidance_embeds=True)
        if guidance is None:
            guidance = torch.full_like(timesteps, 3.5)
        guidance = guidance.repeat_interleave(V)

        pred = self.transformer(
            hidden_states=packed,
            encoder_hidden_states=t5,
            pooled_projections=pooled,
            timestep=t,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]

        # pred: (B*V, S, token_dim) → extract target slot
        pred = pred.view(batch_size, V, *pred.shape[1:])
        return pred[:, target_slot]

    # ------------------------------------------------------------------
    # Conditioning dropout (same logic as RelightSD)
    # ------------------------------------------------------------------

    @staticmethod
    def sample_cond_dropout(
        batch_size: int,
        device: torch.device,
        both_kept: float = 0.85,
        one_dropped: float = 0.10,
        both_dropped: float = 0.05,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r = torch.rand(batch_size, device=device)
        ref1_keep = torch.ones(batch_size, dtype=torch.bool, device=device)
        ref2_keep = torch.ones(batch_size, dtype=torch.bool, device=device)

        both_mask = r < both_dropped
        ref1_keep[both_mask] = False
        ref2_keep[both_mask] = False

        one_mask = (r >= both_dropped) & (r < both_dropped + one_dropped)
        if one_mask.any():
            which = torch.rand(batch_size, device=device) < 0.5
            ref1_keep[one_mask & which] = False
            ref2_keep[one_mask & ~which] = False

        return ref1_keep, ref2_keep

    @staticmethod
    def sample_slot_order(randomize: bool = True) -> list[int]:
        if not randomize:
            return [0, 1, 2]
        return torch.randperm(3).tolist()

    # ------------------------------------------------------------------
    # Training step (flow matching)
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: dict,
        *,
        both_kept: float = 0.85,
        one_dropped: float = 0.10,
        both_dropped: float = 0.05,
        text_drop_prob: float = 0.10,
        randomize_slots: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Single-target training step with flow matching loss."""
        ref1_img = batch["ref1_img"].to(self.device)
        ref2_img = batch["ref2_img"].to(self.device)
        target_img = batch["target_img"].to(self.device)
        pl_ref1 = batch["plucker_ref1"].to(self.device)
        pl_ref2 = batch["plucker_ref2"].to(self.device)
        pl_tgt = batch["plucker_tgt"].to(self.device)
        b = target_img.shape[0]

        ref1_lat = self.encode_image(ref1_img)
        ref2_lat = self.encode_image(ref2_img)
        tgt_lat_clean = self.encode_image(target_img)

        # Offload VAE to CPU — it's not needed during the forward/backward pass
        # and freeing it recovers ~300MB of VRAM for gradient computation.
        vae_device = next(self.vae.parameters()).device
        if vae_device.type == "cuda":
            self.vae.to("cpu")
            torch.cuda.empty_cache()

        _, _, h, w = tgt_lat_clean.shape

        # Flow matching: sample timestep sigma ∈ [0, 1]
        # z_t = (1 - sigma) * x_0 + sigma * noise
        # velocity target = noise - x_0
        noise = torch.randn_like(tgt_lat_clean)

        # Sample timesteps: use scheduler's sigmas for proper distribution
        # For training, sample uniformly in [0, 1]
        sigma = torch.rand(b, device=self.device, dtype=tgt_lat_clean.dtype)
        sigma_expanded = sigma.view(b, 1, 1, 1)

        tgt_lat_noisy = (1.0 - sigma_expanded) * tgt_lat_clean + sigma_expanded * noise
        velocity_target = noise - tgt_lat_clean

        # Text conditioning with dropout
        captions = batch.get("caption", [""] * b)
        if isinstance(captions, str):
            captions = [captions] * b
        captions = list(captions)

        n_text_dropped = 0
        if text_drop_prob > 0:
            for i in range(b):
                if torch.rand(1).item() < text_drop_prob:
                    captions[i] = ""
                    n_text_dropped += 1

        has_text = any(c != "" for c in captions)
        if has_text:
            clip_pooled, t5_hidden = self.get_text_embeddings(captions)
        else:
            clip_pooled = self.null_clip_pooled.expand(b, -1)
            t5_hidden = self.null_t5_emb.expand(b, -1, -1)

        # Reference dropout
        ref1_keep, ref2_keep = self.sample_cond_dropout(
            b, self.device, both_kept, one_dropped, both_dropped,
        )

        slot_order = self.sample_slot_order(randomize=randomize_slots) if self.training else None

        packed, target_slot = self._pack_views(
            ref1_lat, ref2_lat, tgt_lat_noisy,
            pl_ref1, pl_ref2, pl_tgt,
            ref1_keep=ref1_keep, ref2_keep=ref2_keep,
            slot_order=slot_order,
            keep_pluckers=True,
        )

        pred = self._predict_target_velocity(
            packed, sigma, clip_pooled, t5_hidden,
            batch_size=b, target_slot=target_slot,
            latent_h=h, latent_w=w,
        )

        # Unpack predicted tokens back to spatial for loss computation
        pred_spatial = _unpack_latents(pred, h, w, self.LATENT_CH)

        loss = F.mse_loss(pred_spatial.float(), velocity_target.float())

        # Move VAE back to GPU for next iteration's encode step
        if vae_device.type == "cuda":
            self.vae.to(vae_device)

        meta = {
            "n_both_kept": int((ref1_keep & ref2_keep).sum()),
            "n_one_dropped": int((ref1_keep ^ ref2_keep).sum()),
            "n_both_dropped": int((~ref1_keep & ~ref2_keep).sum()),
            "n_text_dropped": n_text_dropped,
        }

        return loss, meta

    # ------------------------------------------------------------------
    # Inference (true CFG)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample(
        self, *,
        ref1_img: torch.Tensor, ref2_img: torch.Tensor,
        pl_ref1: torch.Tensor, pl_ref2: torch.Tensor, pl_tgt: torch.Tensor,
        prompt: str = "", num_steps: int = 28,
        cfg_scale: float = 3.0, cfg_text: float = 3.0,
        seed: int | list[int] = 42,
    ) -> torch.Tensor:
        """Inference with dual CFG via true multi-pass guidance.

        Uses the same dual-CFG strategy as RelightSD:

          v_uncond  = f(no refs, no text)
          v_geo     = f(refs,    no text)
          v_full    = f(refs,    text)

          v = v_uncond
              + cfg_scale * (v_geo  - v_uncond)   # geometry guidance
              + cfg_text  * (v_full - v_geo)       # text guidance

        Returns:
            (B, 3, H, W) decoded images in [-1, 1].
        """
        seeds = [seed] if isinstance(seed, int) else list(seed)
        n_seeds = len(seeds)

        ref1_lat = self.encode_image(ref1_img.to(self.device))
        ref2_lat = self.encode_image(ref2_img.to(self.device))
        pl_ref1 = pl_ref1.to(self.device)
        pl_ref2 = pl_ref2.to(self.device)
        pl_tgt = pl_tgt.to(self.device)

        if n_seeds > 1 and ref1_lat.shape[0] == 1:
            ref1_lat = ref1_lat.expand(n_seeds, -1, -1, -1)
            ref2_lat = ref2_lat.expand(n_seeds, -1, -1, -1)
            pl_ref1 = pl_ref1.expand(n_seeds, -1, -1, -1)
            pl_ref2 = pl_ref2.expand(n_seeds, -1, -1, -1)
            pl_tgt = pl_tgt.expand(n_seeds, -1, -1, -1)

        b = ref1_lat.shape[0]
        _, _, h, w = ref1_lat.shape
        V = self.NUM_VIEWS

        clip_cond, t5_cond = self.get_text_embeddings([prompt])
        clip_cond = clip_cond.expand(b, -1)
        t5_cond = t5_cond.expand(b, -1, -1)
        clip_uncond = self.null_clip_pooled.expand(b, -1)
        t5_uncond = self.null_t5_emb.expand(b, -1, -1)

        # Set up scheduler
        sched = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config)
        sched.set_timesteps(num_steps)

        # Generate independent noise per seed
        latent_shape = (1, self.LATENT_CH, h, w)
        noise_parts = []
        for s in seeds:
            g = torch.Generator(device=self.device).manual_seed(s)
            noise_parts.append(
                torch.randn(latent_shape, generator=g, device=self.device, dtype=ref1_lat.dtype)
            )
        latent = torch.cat(noise_parts, dim=0)

        ref1_drop = torch.zeros(b, dtype=torch.bool, device=self.device)
        ref2_drop = torch.zeros(b, dtype=torch.bool, device=self.device)

        for t in sched.timesteps:
            sigma = torch.full((b,), float(t), device=self.device, dtype=ref1_lat.dtype)

            # Pack 3 CFG branches
            packed_full, tgt_slot = self._pack_views(
                ref1_lat, ref2_lat, latent, pl_ref1, pl_ref2, pl_tgt,
            )
            packed_uncond, _ = self._pack_views(
                ref1_lat, ref2_lat, latent, pl_ref1, pl_ref2, pl_tgt,
                ref1_keep=ref1_drop, ref2_keep=ref2_drop,
                keep_pluckers=False,
            )

            # Batch all 3 CFG branches
            packed_cfg = torch.cat([packed_full, packed_full, packed_uncond], dim=0)
            sigma_cfg = sigma.repeat(3)
            clip_cfg = torch.cat([clip_cond, clip_uncond, clip_uncond], dim=0)
            t5_cfg = torch.cat([t5_cond, t5_uncond, t5_uncond], dim=0)

            pred_all = self._predict_target_velocity(
                packed_cfg, sigma_cfg, clip_cfg, t5_cfg,
                batch_size=3 * b, target_slot=tgt_slot,
                latent_h=h, latent_w=w,
            )

            # Split and apply dual CFG
            pred_chunks = pred_all.view(3, b, *pred_all.shape[1:])
            v_full = _unpack_latents(pred_chunks[0], h, w, self.LATENT_CH)
            v_geo = _unpack_latents(pred_chunks[1], h, w, self.LATENT_CH)
            v_uncond = _unpack_latents(pred_chunks[2], h, w, self.LATENT_CH)

            v = (v_uncond
                 + cfg_scale * (v_geo - v_uncond)
                 + cfg_text * (v_full - v_geo))

            # Euler step: x_{t-dt} = x_t + dt * velocity
            # The scheduler handles this
            latent = sched.step(v, t, latent, return_dict=False)[0]

        return self.decode_latent(latent)
