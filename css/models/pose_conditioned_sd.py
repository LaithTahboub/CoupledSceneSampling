from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from css.models.apg import AdaptiveProjectedGuidance
from css.models.pose_sd_checkpoint import EMAModel, load_pose_sd_checkpoint, save_pose_sd_checkpoint



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



# channel-concat cat3d-style conditioning (no cross-view attention)
class PoseConditionedSD(nn.Module):
    """
    experiment 1: no cross-view attention.

    we concatenate three "views" along channel axis into a single unet input:
      [ref1_lat, pl1, m1, ref2_lat, pl2, m2, tgt_lat_noisy, plt, mt]
    where per-view channels = 4(latent) + 6(plucker) + 1(mask) = 11,
    and total channels = 3 * 11 = 33.

    - refs are clean, target is noisy.
    - unet predicts epsilon for the target latent (still 4 channels output).
    - loss is mse(eps_pred, noise) on the target noise (standard ddpm training).
    - cfg unconditional branch: null text + zeroed ref blocks (lat+pl+mask).
    """

    def __init__(self, pretrained_model: str = "manojb/stable-diffusion-2-1-base", device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)

        self.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.latent_ch = 4
        self.plucker_ch = 6
        self.mask_ch = 1
        self.per_view_ch = self.latent_ch + self.plucker_ch + self.mask_ch  # 11
        self.num_views = 3
        self.in_ch = self.num_views * self.per_view_ch  # 33

        expand_conv_in(self.unet, new_in_channels=self.in_ch)
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

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.unet.parameters() if p.requires_grad]

    def _cache_null_embeddings(self) -> None:
        with torch.no_grad():
            tokens = self.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            self.null_text_emb = self.text_encoder(tokens.input_ids.to(self.device))[0]
            self._text_emb_cache: dict[str, torch.Tensor] = {"": self.null_text_emb}

    @torch.no_grad()
    def get_text_embeddings(self, prompts) -> torch.Tensor:
        normalized = [p if p is not None else "" for p in prompts]
        missing = [p for p in dict.fromkeys(normalized) if p not in self._text_emb_cache]
        if missing:
            tokens = self.tokenizer(missing, padding="max_length", max_length=77, return_tensors="pt")
            embeds = self.text_encoder(tokens.input_ids.to(self.device))[0]
            for i, p in enumerate(missing):
                self._text_emb_cache[p] = embeds[i : i + 1]
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

    def _pack_cat33(
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
        returns x: (b, 33, h, w)

        complicated bit: conditional dropout must make the *entire* ref blocks
        disappear in a way that matches inference-time uncond exactly. so we
        multiply (ref_lat, ref_plucker, ref_mask) by keep ∈ {0,1}.
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
            keep = ref_keep_mask.to(device=device, dtype=dtype)
            keep = rearrange(keep, "b -> b 1 1 1")  # broadcast over c,h,w
            ref1_lat = ref1_lat * keep
            ref2_lat = ref2_lat * keep
            pl1 = pl1 * keep
            pl2 = pl2 * keep
            m1 = m1 * keep
            m2 = m2 * keep

        v1 = torch.cat([ref1_lat, pl1, m1], dim=1)  # (b,11,h,w)
        v2 = torch.cat([ref2_lat, pl2, m2], dim=1)
        vt = torch.cat([tgt_lat,  plt, mt], dim=1)

        x = torch.cat([v1, v2, vt], dim=1)          # (b,33,h,w)
        return x

    def training_step(
        self,
        batch: dict,
        prompt: str | list[str] = "",
        cond_drop_prob: float = 0.1,
        min_timestep: int = 0,
        max_timestep: int | None = None,
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

        x = self._pack_cat33(ref1_lat, ref2_lat, tgt_lat_noisy, pl1, pl2, plt, ref_keep_mask=ref_keep_mask)
        eps = self.unet(x, timesteps, encoder_hidden_states=text_emb).sample  # (b,4,h,w)

        return F.mse_loss(eps, noise)

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

        self.scheduler.set_timesteps(num_steps)

        if target is not None:
            tgt_lat = self.encode_image(target.to(self.device))
            noise = torch.randn_like(tgt_lat)
            start_t = max(0, min(int(self.scheduler.config.num_train_timesteps) - 1, int(start_t)))
            t0 = torch.full((b,), start_t, device=self.device, dtype=torch.long)
            latent = self.scheduler.add_noise(tgt_lat, noise, t0)
            timesteps = self.scheduler.timesteps[self.scheduler.timesteps <= start_t]
        else:
            latent = torch.randn_like(ref1_lat)
            timesteps = self.scheduler.timesteps

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

            if use_cfg or use_apg_guidance:
                x_cond = self._pack_cat33(ref1_lat, ref2_lat, latent, pl1, pl2, plt, ref_keep_mask=None)
                eps_cond = self.unet(x_cond, t_b, encoder_hidden_states=text_cond).sample

                # unconditional: zero refs (lat+pl+mask), keep same noisy target + target plucker
                keep_none = torch.zeros((b,), device=self.device, dtype=torch.bool)
                x_uncond = self._pack_cat33(ref1_lat, ref2_lat, latent, pl1, pl2, plt, ref_keep_mask=keep_none)
                eps_uncond = self.unet(x_uncond, t_b, encoder_hidden_states=text_uncond).sample

                if use_apg_guidance and apg is not None:
                    eps = apg.guide(pred_cond=eps_cond, pred_uncond=eps_uncond)
                else:
                    eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                x = self._pack_cat33(ref1_lat, ref2_lat, latent, pl1, pl2, plt, ref_keep_mask=None)
                eps = self.unet(x, t_b, encoder_hidden_states=text_cond).sample

            latent = self.scheduler.step(eps, t, latent).prev_sample

        return self.decode_latent(latent)
