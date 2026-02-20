"""Pose-Conditioned Stable Diffusion with reference cross-attention.

Architecture:
- UNet: standard 4ch input (noisy latent only), pretrained SD 2.1
- ReferenceEncoder: encodes (ref_latent + plucker_ref) into cross-attention tokens
- Plucker rays are target-anchored: refs' poses are expressed relative to target
- Cross-attention context: [text_tokens, ref1_tokens, ref2_tokens]
- CFG: unconditional = null text + zero ref tokens (truly unconditional)
"""

from os import PathLike
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm


class ReferenceEncoder(nn.Module):
    """Encode (reference latent + Plucker) into cross-attention tokens.

    Input: ref_latent (B, 4, H, W) + plucker (B, 6, H, W) at latent resolution.
    Output: tokens (B, N, output_dim) where N = (H/4)*(W/4).
    """

    def __init__(self, latent_dim=4, plucker_dim=6, hidden_dim=768, output_dim=1024):
        super().__init__()
        in_channels = latent_dim + plucker_dim

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, output_dim, 1),
        )

    def forward(self, ref_latent: torch.Tensor, plucker_ref: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ref_latent, plucker_ref], dim=1)
        feat = self.net(x)
        return feat.flatten(2).permute(0, 2, 1)


class PoseConditionedSD(nn.Module):
    """Stable Diffusion with two-reference cross-attention conditioning.

    Target-anchored design: Plucker rays express reference poses relative to
    the target view. The UNet input is the standard 4-channel noisy latent
    (no pose concat), preserving the pretrained SD initialization.
    """

    def __init__(self, pretrained_model: str = "manojb/stable-diffusion-2-1-base", device: str = "cuda"):
        super().__init__()
        self.device = device

        self.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet").to(device)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.ref_encoder = ReferenceEncoder().to(device)
        self._cache_null_embeddings()

    def configure_trainable(self, unet_train_mode: str = "full") -> None:
        if unet_train_mode == "full":
            self.unet.requires_grad_(True)
            return
        if unet_train_mode != "cond":
            raise ValueError(f"Unknown unet_train_mode: {unet_train_mode}")
        self.unet.requires_grad_(False)
        self.unet.conv_in.requires_grad_(True)
        for name, module in self.unet.named_modules():
            if "attn2" in name:
                module.requires_grad_(True)

    def _cache_null_embeddings(self):
        with torch.no_grad():
            tokens = self.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            self.null_text_emb = self.text_encoder(tokens.input_ids.to(self.device))[0]
            self._text_emb_cache: dict[str, torch.Tensor] = {"": self.null_text_emb}

    @torch.no_grad()
    def get_text_embedding(self, prompt: str) -> torch.Tensor:
        return self.get_text_embeddings([prompt])[0:1]

    @torch.no_grad()
    def get_text_embeddings(self, prompts: Sequence[str]) -> torch.Tensor:
        normalized = [p if p is not None else "" for p in prompts]
        if len(normalized) == 0:
            raise ValueError("prompts cannot be empty")
        missing = [p for p in dict.fromkeys(normalized) if p not in self._text_emb_cache]
        if missing:
            tokens = self.tokenizer(missing, padding="max_length", max_length=77, return_tensors="pt")
            embeds = self.text_encoder(tokens.input_ids.to(self.device))[0]
            for i, p in enumerate(missing):
                self._text_emb_cache[p] = embeds[i : i + 1]
        return torch.cat([self._text_emb_cache[p] for p in normalized], dim=0)

    @torch.no_grad()
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample

    def _encode_refs(
        self,
        ref1_latent: torch.Tensor,
        ref2_latent: torch.Tensor,
        plucker_ref1: torch.Tensor,
        plucker_ref2: torch.Tensor,
    ) -> torch.Tensor:
        ref1_tokens = self.ref_encoder(ref1_latent, plucker_ref1)
        ref2_tokens = self.ref_encoder(ref2_latent, plucker_ref2)
        return torch.cat([ref1_tokens, ref2_tokens], dim=1)

    def build_conditioning_contexts(
        self,
        ref1_latent: torch.Tensor,
        ref2_latent: torch.Tensor,
        plucker_ref1: torch.Tensor,
        plucker_ref2: torch.Tensor,
        text_cond: torch.Tensor,
        text_uncond: torch.Tensor | None = None,
        ref_keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ref_tokens = self._encode_refs(ref1_latent, ref2_latent, plucker_ref1, plucker_ref2)
        if ref_keep_mask is not None:
            keep = ref_keep_mask.to(ref_tokens.dtype).view(ref_tokens.shape[0], 1, 1)
            ref_tokens = ref_tokens * keep

        cond_context = torch.cat([text_cond, ref_tokens], dim=1)
        if text_uncond is None:
            return cond_context, None

        uncond_context = torch.cat([text_uncond, torch.zeros_like(ref_tokens)], dim=1)
        return cond_context, uncond_context

    def forward(
        self,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
        ref1_latent: torch.Tensor,
        ref2_latent: torch.Tensor,
        plucker_ref1: torch.Tensor,
        plucker_ref2: torch.Tensor,
        text_emb: torch.Tensor,
        ref_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond_context, _ = self.build_conditioning_contexts(
            ref1_latent, ref2_latent, plucker_ref1, plucker_ref2,
            text_cond=text_emb, ref_keep_mask=ref_keep_mask,
        )
        return self.unet(noisy_target, timesteps, encoder_hidden_states=cond_context).sample

    def training_step(
        self,
        batch: dict,
        prompt: str | Sequence[str] = "",
        cond_drop_prob: float = 0.1,
        min_timestep: int = 0,
        max_timestep: int | None = None,
    ) -> torch.Tensor:
        ref1_img = batch["ref1_img"].to(self.device)
        ref2_img = batch["ref2_img"].to(self.device)
        target_img = batch["target_img"].to(self.device)
        plucker_ref1 = batch["plucker_ref1"].to(self.device)
        plucker_ref2 = batch["plucker_ref2"].to(self.device)

        B = target_img.shape[0]

        ref1_latent = self.encode_image(ref1_img)
        ref2_latent = self.encode_image(ref2_img)
        target_latent = self.encode_image(target_img)

        noise = torch.randn_like(target_latent)
        total_steps = self.scheduler.config.num_train_timesteps
        lo = max(0, min_timestep)
        hi = total_steps - 1 if max_timestep is None else min(total_steps - 1, max_timestep)
        if hi < lo:
            raise ValueError(f"Invalid timestep range [{lo}, {hi}]")
        timesteps = torch.randint(lo, hi + 1, (B,), device=self.device)
        noisy_target = self.scheduler.add_noise(target_latent, noise, timesteps)

        if isinstance(prompt, str):
            text_emb = self.get_text_embedding(prompt).expand(B, -1, -1)
        else:
            prompt_list = list(prompt)
            if len(prompt_list) != B:
                raise ValueError(f"Prompt batch size mismatch: got {len(prompt_list)} prompts for batch size {B}")
            text_emb = self.get_text_embeddings(prompt_list)

        drop_mask = None
        if self.training and cond_drop_prob > 0:
            drop_mask = torch.rand(B, device=self.device) < cond_drop_prob
            if drop_mask.any():
                num_drop = int(drop_mask.sum().item())
                text_emb = text_emb.clone()
                text_emb[drop_mask] = self.null_text_emb.expand(num_drop, -1, -1)
                ref1_latent = ref1_latent.clone()
                ref2_latent = ref2_latent.clone()
                plucker_ref1 = plucker_ref1.clone()
                plucker_ref2 = plucker_ref2.clone()
                ref1_latent[drop_mask] = 0
                ref2_latent[drop_mask] = 0
                plucker_ref1[drop_mask] = 0
                plucker_ref2[drop_mask] = 0

        loss = F.mse_loss(
            self.forward(
                noisy_target, timesteps,
                ref1_latent, ref2_latent,
                plucker_ref1, plucker_ref2,
                text_emb,
                ref_keep_mask=(None if drop_mask is None else ~drop_mask),
            ),
            noise,
        )
        return loss

    @torch.inference_mode()
    def sample(
        self,
        ref1_img: torch.Tensor,
        ref2_img: torch.Tensor,
        plucker_ref1: torch.Tensor,
        plucker_ref2: torch.Tensor,
        prompt: str = "",
        num_steps: int = 50,
        cfg_scale: float = 7.5,
        target: torch.Tensor | None = None,
        start_t: int = 1000,
    ) -> torch.Tensor:
        B = ref1_img.shape[0]

        ref1_latent = self.encode_image(ref1_img.to(self.device))
        ref2_latent = self.encode_image(ref2_img.to(self.device))
        plucker_ref1 = plucker_ref1.to(self.device)
        plucker_ref2 = plucker_ref2.to(self.device)

        text_cond = self.get_text_embedding(prompt).expand(B, -1, -1)
        null_text_batch = self.null_text_emb.expand(B, -1, -1)
        cond_context, uncond_context = self.build_conditioning_contexts(
            ref1_latent, ref2_latent, plucker_ref1, plucker_ref2,
            text_cond=text_cond, text_uncond=null_text_batch,
        )

        self.scheduler.set_timesteps(num_steps)

        if target is not None:
            target_latent = self.encode_image(target.to(self.device))
            noise = torch.randn_like(target_latent)
            start_t = max(0, min(int(self.scheduler.config.num_train_timesteps) - 1, int(start_t)))
            timestep = torch.full((B,), start_t, device=self.device, dtype=torch.long)
            latent = self.scheduler.add_noise(target_latent, noise, timestep)
            timesteps = self.scheduler.timesteps[self.scheduler.timesteps <= start_t]
        else:
            latent = torch.randn_like(ref1_latent)
            timesteps = self.scheduler.timesteps

        use_cfg = cfg_scale > 1.0 and uncond_context is not None

        for t in tqdm(timesteps, desc="Sampling"):
            t_val = int(t.item()) if isinstance(t, torch.Tensor) else int(t)

            if use_cfg:
                latent_pair = torch.cat([latent, latent], dim=0)
                context_pair = torch.cat([uncond_context, cond_context], dim=0)
                t_pair = torch.full((2 * B,), t_val, device=self.device, dtype=torch.long)
                eps_pair = self.unet(latent_pair, t_pair, encoder_hidden_states=context_pair).sample
                eps_uncond, eps_cond = eps_pair.chunk(2)
                eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                t_batch = torch.full((B,), t_val, device=self.device, dtype=torch.long)
                eps = self.unet(latent, t_batch, encoder_hidden_states=cond_context).sample

            latent = self.scheduler.step(eps, t, latent).prev_sample

        return self.decode_latent(latent)


def save_pose_sd_checkpoint(model: PoseConditionedSD, checkpoint_path: str | PathLike) -> None:
    torch.save(
        {
            "unet": model.unet.state_dict(),
            "ref_encoder": model.ref_encoder.state_dict(),
        },
        checkpoint_path,
    )


def load_pose_sd_checkpoint(model: PoseConditionedSD, checkpoint_path: str | PathLike, device: str) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "unet" in state:
        model.unet.load_state_dict(state["unet"])
        model.ref_encoder.load_state_dict(state["ref_encoder"])
    else:
        model.unet.load_state_dict(state)
