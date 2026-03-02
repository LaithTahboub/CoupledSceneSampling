from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from css.models.apg import AdaptiveProjectedGuidance
from css.models.cross_view_attention import CrossViewAttention



# unet conv_in expansion
def expand_conv_in(unet: UNet2DConditionModel, new_in_channels: int) -> None:
    """
    expand unet.conv_in to accept new_in_channels.

    keeps pretrained sd weights for the first old.in_channels (usually 4),
    and zero-inits the extra channels so the model starts "as sd" when the new
    conditioning channels are near-zero.
    """
    old = unet.conv_in
    if old.in_channels == new_in_channels:
        return

    new = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=(old.bias is not None),
    ).to(device=old.weight.device, dtype=old.weight.dtype)

    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, : old.in_channels].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)

    unet.conv_in = new
    unet.config.in_channels = new_in_channels



class PoseConditionedSD(nn.Module):
    """
    CAT3D-style multi-view conditioning with cross-view attention.

    Input views are packed as `(batch * 3, 11, h, w)` in fixed order:
      [target, ref1, ref2]
    with per-view channels:
      [latent(4), plucker(6), mask(1)].

    - refs are clean; target is noisy.
    - cross-view attention lets self-attention exchange tokens across views.
    - unet predicts epsilon for the target latent (still 4 channels output).
    - loss is mse(eps_pred, noise) on the target noise (standard ddpm training).
    - cfg unconditional branch: null text + zeroed ref view blocks.
    """

    def __init__(self, pretrained_model: str = "manojb/stable-diffusion-2-1-base", device: str = "cuda"):
        super().__init__()
        if device == "cuda" and not torch.cuda.is_available():
            print("[pose-sd] CUDA not available, falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        self.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
        self.inference_scheduler = EulerDiscreteScheduler.from_config(self.scheduler.config)
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.eval()

        self.latent_ch = 4
        self.plucker_ch = 6
        self.mask_ch = 1
        self.per_view_ch = self.latent_ch + self.plucker_ch + self.mask_ch  # 11
        self.num_views = 3
        self.in_ch = self.per_view_ch

        expand_conv_in(self.unet, new_in_channels=self.in_ch)
        self.cross_view_attention = CrossViewAttention(num_views=self.num_views, include_high_res=False)
        self.cross_view_attention.attach(self.unet)

        self._text_cache_limit = 4096
        self._cache_null_embeddings()

    def configure_trainable(self, unet_train_mode: str = "full") -> None:
        if unet_train_mode == "full":
            self.unet.requires_grad_(True)
            return
        if unet_train_mode != "cond":
            raise ValueError(f"unknown unet_train_mode: {unet_train_mode}")

        # "cond" for this experiment means: train conv_in + attention/cross-attn.
        self.unet.requires_grad_(False)
        self.unet.conv_in.requires_grad_(True)
        for name, p in self.unet.named_parameters():
            if "attn" in name:
                p.requires_grad_(True)

    def configure_memory_optimizations(
        self,
        gradient_checkpointing: bool = False,
        xformers_attention: bool = False,
    ) -> None:
        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
        if xformers_attention:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as exc:  # pragma: no cover - env-dependent
                print(f"[pose-sd] xformers attention unavailable: {exc}")
        # xformers can swap processors; reattach cross-view wrappers afterwards.
        self.cross_view_attention.attach(self.unet)

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.unet.parameters() if p.requires_grad]

    def _cache_null_embeddings(self) -> None:
        with torch.no_grad():
            tokens = self.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            self.null_text_emb = self.text_encoder(tokens.input_ids.to(self.device))[0]
            self._text_emb_cache: dict[str, torch.Tensor] = {"": self.null_text_emb}

    @torch.no_grad()
    def get_text_embeddings(self, prompts: Sequence[str | None]) -> torch.Tensor:
        normalized = [p if p is not None else "" for p in prompts]
        missing = [p for p in dict.fromkeys(normalized) if p not in self._text_emb_cache]
        if missing:
            tokens = self.tokenizer(missing, padding="max_length", max_length=77, return_tensors="pt")
            embeds = self.text_encoder(tokens.input_ids.to(self.device))[0]
            for i, p in enumerate(missing):
                self._text_emb_cache[p] = embeds[i : i + 1]
            if len(self._text_emb_cache) > self._text_cache_limit:
                self._text_emb_cache = {"": self.null_text_emb}
        return torch.cat([self._text_emb_cache[p] for p in normalized], dim=0)

    @torch.no_grad()
    def get_text_embedding(self, prompt: str) -> torch.Tensor:
        return self.get_text_embeddings([prompt])[0:1]

    @torch.no_grad()
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample

    def _pack_views(
        self,
        ref1_lat: torch.Tensor,
        ref2_lat: torch.Tensor,
        tgt_lat: torch.Tensor,
        pl1: torch.Tensor,
        pl2: torch.Tensor,
        plt: torch.Tensor,
        ref_keep_mask: torch.Tensor | None = None,  # (b,) bool, true=keep refs
    ) -> torch.Tensor:
        """
        returns packed views: (b * 3, 11, h, w) in order [target, ref1, ref2]

        conditional dropout must hide full reference blocks in both training
        and CFG-unconditional inference (lat + plucker + mask).
        """
        b, _, h, w = tgt_lat.shape
        dtype = tgt_lat.dtype
        device = tgt_lat.device

        ones = torch.ones((b, 1, h, w), device=device, dtype=dtype)
        zeros = torch.zeros((b, 1, h, w), device=device, dtype=dtype)

        m1 = ones
        m2 = ones
        mt = zeros

        if ref_keep_mask is not None:
            keep = ref_keep_mask.to(device=device, dtype=dtype).view(b, 1, 1, 1)
            ref1_lat = ref1_lat * keep
            ref2_lat = ref2_lat * keep
            pl1 = pl1 * keep
            pl2 = pl2 * keep
            m1 = m1 * keep
            m2 = m2 * keep

        v1 = torch.cat([ref1_lat, pl1, m1], dim=1)
        v2 = torch.cat([ref2_lat, pl2, m2], dim=1)
        vt = torch.cat([tgt_lat, plt, mt], dim=1)

        views = torch.stack([vt, v1, v2], dim=1)  # (b, 3, 11, h, w)
        return views.reshape(b * self.num_views, self.per_view_ch, h, w)

    def _repeat_text_for_views(self, text_emb: torch.Tensor) -> torch.Tensor:
        return text_emb.repeat_interleave(self.num_views, dim=0)

    def _extract_target_view(self, pred_views: torch.Tensor, batch_size: int) -> torch.Tensor:
        if pred_views.shape[0] != batch_size * self.num_views:
            raise ValueError(
                f"unexpected view-packed batch: got {pred_views.shape[0]}, "
                f"expected {batch_size * self.num_views}",
            )
        return pred_views.view(batch_size, self.num_views, *pred_views.shape[1:])[:, 0]

    def _predict_target_eps(
        self,
        packed_views: torch.Tensor,
        timesteps: torch.Tensor,
        text_emb: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        t_views = timesteps.repeat_interleave(self.num_views)
        text_views = self._repeat_text_for_views(text_emb)
        pred_views = self.unet(packed_views, t_views, encoder_hidden_states=text_views).sample
        return self._extract_target_view(pred_views, batch_size)

    def training_step(
        self,
        batch: dict,
        prompt: str | list[str] = "",
        cond_drop_prob: float = 0.1,
        min_timestep: int = 0,
        max_timestep: int | None = None,
        noise_offset: float = 0.0,
        min_snr_gamma: float | None = None,
    ) -> torch.Tensor:
        ref1_img = batch["ref1_img"].to(self.device)
        ref2_img = batch["ref2_img"].to(self.device)
        target_img = batch["target_img"].to(self.device)

        # all pluckers are ref1-anchored, including target
        pl1 = batch["plucker_ref1"].to(self.device)
        pl2 = batch["plucker_ref2"].to(self.device)
        plt = batch["plucker_tgt"].to(self.device)

        b = target_img.shape[0]

        ref1_lat = self.encode_image(ref1_img)
        ref2_lat = self.encode_image(ref2_img)
        tgt_lat_clean = self.encode_image(target_img)

        noise = torch.randn_like(tgt_lat_clean)
        if noise_offset > 0:
            noise = noise + noise_offset * torch.randn(
                (b, noise.shape[1], 1, 1),
                device=self.device,
                dtype=noise.dtype,
            )

        total = int(self.scheduler.config.num_train_timesteps)
        lo = max(0, int(min_timestep))
        hi = total - 1 if max_timestep is None else min(total - 1, int(max_timestep))
        if hi < lo:
            raise ValueError(f"invalid timestep range [{lo}, {hi}]")

        timesteps = torch.randint(lo, hi + 1, (b,), device=self.device, dtype=torch.long)
        tgt_lat_noisy = self.scheduler.add_noise(tgt_lat_clean, noise, timesteps)

        # text embeddings
        if isinstance(prompt, str):
            text_emb = self.get_text_embedding(prompt).expand(b, -1, -1)
        else:
            if len(prompt) != b:
                raise ValueError(f"prompt batch size mismatch: got {len(prompt)} prompts for batch size {b}")
            text_emb = self.get_text_embeddings(prompt)

        # cfg-style dropout: drop text + refs together
        ref_keep_mask = None
        if self.training and cond_drop_prob > 0:
            drop = (torch.rand(b, device=self.device) < cond_drop_prob)
            if drop.any():
                text_emb = text_emb.clone()
                text_emb[drop] = self.null_text_emb.expand(int(drop.sum().item()), -1, -1)
                ref_keep_mask = ~drop

        packed_views = self._pack_views(
            ref1_lat, ref2_lat, tgt_lat_noisy, pl1, pl2, plt, ref_keep_mask=ref_keep_mask,
        )
        pred = self._predict_target_eps(packed_views, timesteps, text_emb, batch_size=b)

        prediction_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(tgt_lat_clean, noise, timesteps)
        else:
            raise ValueError(f"Unsupported scheduler prediction type: {prediction_type}")

        loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=(1, 2, 3))

        if min_snr_gamma is not None and min_snr_gamma > 0:
            alphas = self.scheduler.alphas_cumprod.to(device=self.device, dtype=loss.dtype)
            alpha_t = alphas[timesteps]
            snr = alpha_t / (1.0 - alpha_t).clamp_min(1e-8)
            gamma = torch.full_like(snr, float(min_snr_gamma))
            if prediction_type == "epsilon":
                weights = torch.minimum(snr, gamma) / snr.clamp_min(1e-8)
            else:  # v_prediction
                weights = torch.minimum(snr, gamma) / (snr + 1.0).clamp_min(1e-8)
            loss = loss * weights

        return loss.mean()

    @torch.inference_mode()
    def sample(
        self,
        ref1_img: torch.Tensor,
        ref2_img: torch.Tensor,
        pl1: torch.Tensor,
        pl2: torch.Tensor,
        plt: torch.Tensor,
        prompt: str = "",
        num_steps: int = 50,
        cfg_scale: float = 7.5,
        target: torch.Tensor | None = None,
        start_t: int = 1000,
        use_apg: bool = False,
        apg_eta: float = 0.0,
        apg_momentum: float = -0.5,
        apg_norm_threshold: float = 0.0,
        apg_eps: float = 1e-12,
    ) -> torch.Tensor:
        b = ref1_img.shape[0]

        ref1_lat = self.encode_image(ref1_img.to(self.device))
        ref2_lat = self.encode_image(ref2_img.to(self.device))
        pl1 = pl1.to(self.device)
        pl2 = pl2.to(self.device)
        plt = plt.to(self.device)

        text_cond = self.get_text_embedding(prompt).expand(b, -1, -1)
        text_uncond = self.null_text_emb.expand(b, -1, -1)

        scheduler = self.scheduler

        scheduler.set_timesteps(num_steps)

        if target is not None:
            tgt_lat = self.encode_image(target.to(self.device))
            noise = torch.randn_like(tgt_lat)
            start_t = max(0, min(int(scheduler.config.num_train_timesteps) - 1, int(start_t)))
            t0 = torch.full((b,), start_t, device=self.device, dtype=torch.long)
            latent = scheduler.add_noise(tgt_lat, noise, t0)
            timesteps = scheduler.timesteps[scheduler.timesteps <= start_t]
        else:
            latent = torch.randn_like(ref1_lat)
            timesteps = scheduler.timesteps

        use_apg_guidance = use_apg and cfg_scale > 1.0
        use_cfg = (not use_apg_guidance) and cfg_scale > 1.0

        apg = None
        if use_apg_guidance:
            apg = AdaptiveProjectedGuidance(
                guidance_scale=cfg_scale,
                eta=apg_eta,
                momentum=apg_momentum,
                norm_threshold=apg_norm_threshold,
                eps=apg_eps,
            )

        for t in tqdm(timesteps, desc="sampling"):
            t_val = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            t_b = torch.full((b,), t_val, device=self.device, dtype=torch.long)

            latent = scheduler.scale_model_input(latent, t)

            if use_cfg or use_apg_guidance:
                x_cond = self._pack_views(ref1_lat, ref2_lat, latent, pl1, pl2, plt, ref_keep_mask=None)
                eps_cond = self._predict_target_eps(x_cond, t_b, text_cond, batch_size=b)

                # unconditional: zero refs (lat+pl+mask), keep same noisy target + target plucker
                keep_none = torch.zeros((b,), device=self.device, dtype=torch.bool)
                x_uncond = self._pack_views(ref1_lat, ref2_lat, latent, pl1, pl2, plt, ref_keep_mask=keep_none)
                eps_uncond = self._predict_target_eps(x_uncond, t_b, text_uncond, batch_size=b)

                if use_apg_guidance and apg is not None:
                    eps = apg.guide(pred_cond=eps_cond, pred_uncond=eps_uncond)
                else:
                    eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                x = self._pack_views(ref1_lat, ref2_lat, latent, pl1, pl2, plt, ref_keep_mask=None)
                eps = self._predict_target_eps(x, t_b, text_cond, batch_size=b)

            latent = scheduler.step(eps, t, latent).prev_sample

        return self.decode_latent(latent)
