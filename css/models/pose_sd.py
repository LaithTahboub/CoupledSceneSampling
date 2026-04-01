"""PoseSD: 3-view model with Plucker ray conditioning.

Views are packed as (B*V, 11, h, w) with per-view channels:
latent(4) + plucker(6) + mask(1).

mask=1 for reference views (clean), mask=0 for target views (noised).

Supports:
- 3-view mode: [target, ref1, ref2]
- Randomized slot order during training
- Variable conditioning count (1 or 2 refs)
- Structured conditioning dropout (independent ref1/ref2/text dropping)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from css.models.cross_view_attention import inflate_unet_attention


def _expand_conv_in(unet: UNet2DConditionModel, new_in_channels: int) -> None:
    old = unet.conv_in
    if old.in_channels == new_in_channels:
        return
    new = nn.Conv2d(
        new_in_channels, old.out_channels,
        kernel_size=old.kernel_size, stride=old.stride,
        padding=old.padding, bias=(old.bias is not None),
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_channels].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.config.in_channels = new_in_channels


class PoseSD(nn.Module):
    """3-view model: 2 refs + 1 target with Plucker conditioning."""

    NUM_VIEWS = 3          # default for single-target (backward compat)
    PER_VIEW_CH = 11       # latent(4) + plucker(6) + mask(1)

    def __init__(self, pretrained_model: str = "manojb/stable-diffusion-2-1-base", device: str = "cuda"):
        super().__init__()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
        self.inference_scheduler = DDIMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

        self.vae.requires_grad_(False).eval()
        self.text_encoder.requires_grad_(False).eval()

        _expand_conv_in(self.unet, self.PER_VIEW_CH)
        inflated = inflate_unet_attention(self.unet, num_views=self.NUM_VIEWS, include_high_res=False)
        print(f"[pose-sd] inflated {len(inflated)} self-attention layers to multi-view")

        with torch.no_grad():
            tokens = self.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            self.null_text_emb = self.text_encoder(tokens.input_ids.to(self.device))[0]

    def configure_trainable(self, train_mode: str = "full") -> None:
        if train_mode == "full":
            self.unet.requires_grad_(True)
            return
        # Legacy "cond" mode: only conv_in + attention layers
        self.unet.requires_grad_(False)
        self.unet.conv_in.requires_grad_(True)
        for name, param in self.unet.named_parameters():
            if "attn" in name:
                param.requires_grad_(True)

    def configure_memory_optimizations(
        self,
        gradient_checkpointing: bool = False,
        xformers_attention: bool = False,
        compile_unet: bool = False,
    ) -> None:
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        if xformers_attention:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                print(f"[pose-sd] xformers unavailable: {exc}")
        if compile_unet:
            mode = "default" if gradient_checkpointing else "reduce-overhead"
            self.unet = torch.compile(self.unet, mode=mode)
            print(f"[pose-sd] compiled UNet with torch.compile(mode='{mode}')")

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.unet.parameters() if p.requires_grad]

    @torch.no_grad()
    def get_text_embeddings(self, prompts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(prompts, padding="max_length", max_length=77, return_tensors="pt")
        return self.text_encoder(tokens.input_ids.to(self.device))[0]

    @torch.no_grad()
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample

    # ------------------------------------------------------------------
    # View packing (single-target, backward compatible)
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
        """Pack 3 views into (B*3, 11, h, w).

        Args:
            keep_pluckers: If True, Plucker ray maps are never zeroed by the
                keep masks — only ref latents and masks are dropped.  This
                preserves the full geometric layout in the unconditional CFG
                branch so the model always knows the 3-camera arrangement.
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

        return torch.stack(ordered, dim=1).reshape(
            b * self.NUM_VIEWS, self.PER_VIEW_CH, h, w,
        ), target_slot

    def _predict_target_eps(
        self, packed: torch.Tensor, timesteps: torch.Tensor,
        text_emb: torch.Tensor, batch_size: int, target_slot: int = 0,
    ) -> torch.Tensor:
        t = timesteps.repeat_interleave(self.NUM_VIEWS)
        te = text_emb.repeat_interleave(self.NUM_VIEWS, dim=0)
        pred = self.unet(packed, t, encoder_hidden_states=te).sample
        return pred.view(batch_size, self.NUM_VIEWS, *pred.shape[1:])[:, target_slot]

    # ------------------------------------------------------------------
    # Conditioning dropout
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
    # Training steps
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
        """Single-target training step."""
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

        noise = torch.randn_like(tgt_lat_clean)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,),
                                  device=self.device, dtype=torch.long)
        tgt_lat_noisy = self.scheduler.add_noise(tgt_lat_clean, noise, timesteps)

        # Text conditioning: use captions from batch with dropout
        captions = batch.get("caption", [""] * b)
        if isinstance(captions, str):
            captions = [captions] * b
        captions = list(captions)

        # Text conditioning dropout: independently replace captions with ""
        n_text_dropped = 0
        if text_drop_prob > 0:
            for i in range(b):
                if torch.rand(1).item() < text_drop_prob:
                    captions[i] = ""
                    n_text_dropped += 1

        # Encode text: use null embedding for empty captions, real embeddings otherwise
        has_text = any(c != "" for c in captions)
        if has_text:
            text_emb = self.get_text_embeddings(captions)
        else:
            text_emb = self.null_text_emb.expand(b, -1, -1)

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
        pred = self._predict_target_eps(packed, timesteps, text_emb, batch_size=b, target_slot=target_slot)

        loss = F.mse_loss(pred.float(), noise.float())

        meta = {
            "n_both_kept": int((ref1_keep & ref2_keep).sum()),
            "n_one_dropped": int((ref1_keep ^ ref2_keep).sum()),
            "n_both_dropped": int((~ref1_keep & ~ref2_keep).sum()),
            "n_text_dropped": n_text_dropped,
        }

        return loss, meta

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample(
        self, *,
        ref1_img: torch.Tensor, ref2_img: torch.Tensor,
        pl_ref1: torch.Tensor, pl_ref2: torch.Tensor, pl_tgt: torch.Tensor,
        prompt: str = "", num_steps: int = 25,
        cfg_scale: float = 3.0, cfg_text: float = 3.0,
        seed: int | list[int] = 42,
    ) -> torch.Tensor:
        """Inference with dual CFG, batched across both CFG branches and seeds.

        All 3 CFG branches are batched into a single UNet forward pass per
        denoising step.  When ``seed`` is a list, each seed produces an
        independent sample in a single batched call — inputs are automatically
        expanded along the batch dimension.

          eps_uncond  = f(no refs, no text)
          eps_geo     = f(refs,    no text)
          eps_full    = f(refs,    text)

          eps = eps_uncond
              + cfg_scale * (eps_geo  - eps_uncond)   # geometry guidance
              + cfg_text  * (eps_full - eps_geo)       # text guidance

        Args:
            seed: single int or list of ints.  List produces len(seed) outputs
                  in one batched forward pass (inputs must have B=1).

        Returns:
            (B, 3, H, W) decoded images in [-1, 1], where B = len(seeds).
        """
        seeds = [seed] if isinstance(seed, int) else list(seed)
        n_seeds = len(seeds)

        ref1_lat = self.encode_image(ref1_img.to(self.device))
        ref2_lat = self.encode_image(ref2_img.to(self.device))
        pl_ref1 = pl_ref1.to(self.device)
        pl_ref2 = pl_ref2.to(self.device)
        pl_tgt = pl_tgt.to(self.device)

        # Expand B=1 inputs to batch of n_seeds
        if n_seeds > 1 and ref1_lat.shape[0] == 1:
            ref1_lat = ref1_lat.expand(n_seeds, -1, -1, -1)
            ref2_lat = ref2_lat.expand(n_seeds, -1, -1, -1)
            pl_ref1 = pl_ref1.expand(n_seeds, -1, -1, -1)
            pl_ref2 = pl_ref2.expand(n_seeds, -1, -1, -1)
            pl_tgt = pl_tgt.expand(n_seeds, -1, -1, -1)

        b = ref1_lat.shape[0]
        V = self.NUM_VIEWS

        text_cond = self.get_text_embeddings([prompt]).expand(b, -1, -1)
        text_uncond = self.null_text_emb.expand(b, -1, -1)

        sched = self.inference_scheduler
        sched.set_timesteps(num_steps)

        # Generate independent noise per seed
        latent_shape = (1, *ref1_lat.shape[1:])
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
            t_b = torch.full((b,), int(t), device=self.device, dtype=torch.long)
            latent_in = sched.scale_model_input(latent, t)

            # Pack views for both CFG branches (full/geo share the same packing)
            packed_full, tgt_slot = self._pack_views(
                ref1_lat, ref2_lat, latent_in, pl_ref1, pl_ref2, pl_tgt,
            )
            packed_uncond, _ = self._pack_views(
                ref1_lat, ref2_lat, latent_in, pl_ref1, pl_ref2, pl_tgt,
                ref1_keep=ref1_drop, ref2_keep=ref2_drop,
                keep_pluckers=False,
            )

            # Batch all 3 CFG branches: [full, geo, uncond] → single UNet call
            packed_cfg = torch.cat([packed_full, packed_full, packed_uncond], dim=0)
            t_cfg = t_b.repeat_interleave(V).repeat(3)
            text_cfg = torch.cat([
                text_cond.repeat_interleave(V, dim=0),
                text_uncond.repeat_interleave(V, dim=0),
                text_uncond.repeat_interleave(V, dim=0),
            ], dim=0)

            pred_all = self.unet(packed_cfg, t_cfg, encoder_hidden_states=text_cfg).sample
            pred_all = pred_all.view(3, b, V, *pred_all.shape[1:])

            eps_full = pred_all[0, :, tgt_slot]
            eps_geo = pred_all[1, :, tgt_slot]
            eps_uncond = pred_all[2, :, tgt_slot]

            eps = (eps_uncond
                   + cfg_scale * (eps_geo - eps_uncond)
                   + cfg_text * (eps_full - eps_geo))
            latent = sched.step(eps, t, latent).prev_sample

        return self.decode_latent(latent)
