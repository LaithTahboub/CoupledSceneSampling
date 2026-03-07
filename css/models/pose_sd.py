"""PoseSD: 3-view CAT3D-style model with Plucker ray conditioning.

Views are packed as (B*3, 11, h, w) in order [target, ref1, ref2] with
per-view channels: latent(4) + plucker(6) + mask(1).

Extends the working SingleRefSD pattern to 3 views with pose conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from css.models.cross_view_attention import CrossViewAttention


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
    """3-view CAT3D-style model: [target, ref1, ref2] with Plucker conditioning."""

    NUM_VIEWS = 3
    PER_VIEW_CH = 11  # latent(4) + plucker(6) + mask(1)

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

        self.vae.requires_grad_(False).eval()
        self.text_encoder.requires_grad_(False).eval()

        _expand_conv_in(self.unet, self.PER_VIEW_CH)
        self.cross_view_attention = CrossViewAttention(num_views=self.NUM_VIEWS, include_high_res=False)
        self.cross_view_attention.attach(self.unet)

        with torch.no_grad():
            tokens = self.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            self.null_text_emb = self.text_encoder(tokens.input_ids.to(self.device))[0]

    def configure_trainable(self, train_mode: str = "cond") -> None:
        if train_mode == "full":
            self.unet.requires_grad_(True)
            return
        self.unet.requires_grad_(False)
        self.unet.conv_in.requires_grad_(True)
        for name, param in self.unet.named_parameters():
            if "attn" in name:
                param.requires_grad_(True)

    def configure_memory_optimizations(
        self, gradient_checkpointing: bool = False, xformers_attention: bool = False,
    ) -> None:
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        if xformers_attention:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                print(f"[pose-sd] xformers unavailable: {exc}")
        self.cross_view_attention.attach(self.unet)

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

    def _pack_views(
        self,
        ref1_lat: torch.Tensor, ref2_lat: torch.Tensor, tgt_lat: torch.Tensor,
        pl_ref1: torch.Tensor, pl_ref2: torch.Tensor, pl_tgt: torch.Tensor,
        ref_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, _, h, w = tgt_lat.shape
        dtype, device = tgt_lat.dtype, tgt_lat.device

        m_ref = torch.ones((b, 1, h, w), device=device, dtype=dtype)
        m_tgt = torch.zeros((b, 1, h, w), device=device, dtype=dtype)

        if ref_keep_mask is not None:
            keep = ref_keep_mask.to(device=device, dtype=dtype).view(b, 1, 1, 1)
            ref1_lat = ref1_lat * keep
            ref2_lat = ref2_lat * keep
            pl_ref1 = pl_ref1 * keep
            pl_ref2 = pl_ref2 * keep
            m_ref = m_ref * keep

        v_tgt = torch.cat([tgt_lat, pl_tgt, m_tgt], dim=1)
        v_r1 = torch.cat([ref1_lat, pl_ref1, m_ref], dim=1)
        v_r2 = torch.cat([ref2_lat, pl_ref2, m_ref], dim=1)

        # Interleave views so cross-view attention groups them correctly:
        # [tgt0, r1_0, r2_0, tgt1, r1_1, r2_1, ...]
        return torch.stack([v_tgt, v_r1, v_r2], dim=1).reshape(
            b * self.NUM_VIEWS, self.PER_VIEW_CH, h, w,
        )

    def _predict_target_eps(
        self, packed: torch.Tensor, timesteps: torch.Tensor,
        text_emb: torch.Tensor, batch_size: int,
    ) -> torch.Tensor:
        t = timesteps.repeat_interleave(self.NUM_VIEWS)
        te = text_emb.repeat_interleave(self.NUM_VIEWS, dim=0)
        pred = self.unet(packed, t, encoder_hidden_states=te).sample
        return pred.view(batch_size, self.NUM_VIEWS, *pred.shape[1:])[:, 0]

    def training_step(self, batch: dict, *, cond_drop_prob: float = 0.15) -> torch.Tensor:
        ref1_img = batch["ref1_img"].to(self.device)
        ref2_img = batch["ref2_img"].to(self.device)
        target_img = batch["target_img"].to(self.device)
        pl_ref1 = batch["plucker_ref1"].to(self.device)
        pl_ref2 = batch["plucker_ref2"].to(self.device)
        pl_tgt = batch["plucker_tgt"].to(self.device)
        prompts = list(batch["prompt"])
        b = target_img.shape[0]

        ref1_lat = self.encode_image(ref1_img)
        ref2_lat = self.encode_image(ref2_img)
        tgt_lat_clean = self.encode_image(target_img)

        noise = torch.randn_like(tgt_lat_clean)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (b,),
                                  device=self.device, dtype=torch.long)
        tgt_lat_noisy = self.scheduler.add_noise(tgt_lat_clean, noise, timesteps)

        text_emb = self.get_text_embeddings(prompts)

        ref_keep_mask = None
        if self.training and cond_drop_prob > 0:
            drop = torch.rand(b, device=self.device) < cond_drop_prob
            if drop.any():
                text_emb = text_emb.clone()
                text_emb[drop] = self.null_text_emb.expand(int(drop.sum()), -1, -1)
                ref_keep_mask = ~drop

        packed = self._pack_views(ref1_lat, ref2_lat, tgt_lat_noisy,
                                  pl_ref1, pl_ref2, pl_tgt, ref_keep_mask)
        pred = self._predict_target_eps(packed, timesteps, text_emb, batch_size=b)

        return F.mse_loss(pred.float(), noise.float())

    @torch.inference_mode()
    def sample(
        self, *,
        ref1_img: torch.Tensor, ref2_img: torch.Tensor,
        pl_ref1: torch.Tensor, pl_ref2: torch.Tensor, pl_tgt: torch.Tensor,
        prompt: str, num_steps: int = 50, cfg_scale: float = 4.0, seed: int = 42,
    ) -> torch.Tensor:
        ref1_lat = self.encode_image(ref1_img.to(self.device))
        ref2_lat = self.encode_image(ref2_img.to(self.device))
        pl_ref1 = pl_ref1.to(self.device)
        pl_ref2 = pl_ref2.to(self.device)
        pl_tgt = pl_tgt.to(self.device)
        b = ref1_lat.shape[0]

        text_cond = self.get_text_embeddings([prompt]).expand(b, -1, -1)
        text_uncond = self.null_text_emb.expand(b, -1, -1)

        self.scheduler.set_timesteps(num_steps)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        latent = torch.randn(ref1_lat.shape, generator=gen, device=self.device, dtype=ref1_lat.dtype)

        for t in self.scheduler.timesteps:
            t_b = torch.full((b,), int(t), device=self.device, dtype=torch.long)
            latent_in = self.scheduler.scale_model_input(latent, t)

            packed_cond = self._pack_views(ref1_lat, ref2_lat, latent_in,
                                           pl_ref1, pl_ref2, pl_tgt)
            eps_cond = self._predict_target_eps(packed_cond, t_b, text_cond, b)

            keep_none = torch.zeros(b, device=self.device, dtype=torch.bool)
            packed_uncond = self._pack_views(ref1_lat, ref2_lat, latent_in,
                                             pl_ref1, pl_ref2, pl_tgt, keep_none)
            eps_uncond = self._predict_target_eps(packed_uncond, t_b, text_uncond, b)

            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            latent = self.scheduler.step(eps, t, latent).prev_sample

        return self.decode_latent(latent)
