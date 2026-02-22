"""Pose-Conditioned Stable Diffusion with reference cross-attention.

Architecture:
- UNet: standard 4ch input (noisy latent only), pretrained SD 2.1
- ReferenceEncoder: encodes (ref_latent + plucker_ref) into cross-attention tokens
- Plucker rays are target-anchored: refs' poses are expressed relative to target
- Cross-attention context: [text_tokens, ref1_tokens, ref2_tokens]
- CFG: unconditional = null text + zero ref tokens (truly unconditional)

Fixes applied:
- Conditioning drop: only mask-based zeroing on encoder output, never zero encoder inputs
- Train/inference unconditional mismatch resolved
- EMA support added
- Checkpoint saves/loads optimizer, LR scheduler, EMA, epoch, step
- Device stored as torch.device for robustness
- cross_attention_dim read from UNet config instead of hardcoded
- configure_trainable iterates named_parameters not named_modules
- get_trainable_parameters helper added
"""

from __future__ import annotations

from os import PathLike
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Reference encoder
# ---------------------------------------------------------------------------

class ReferenceEncoder(nn.Module):
    """Encode (reference latent + Plucker) into cross-attention tokens.

    Input: ref_latent (B, 4, H, W) + plucker (B, 6, H, W) at latent resolution.
    Output: tokens (B, N, output_dim) where N = (H/4)*(W/4).
    """

    def __init__(self, latent_dim: int = 4, plucker_dim: int = 6, hidden_dim: int = 768, output_dim: int = 1024):
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


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, parameters: list[nn.Parameter], decay: float = 0.9999):
        self.decay = decay
        self.shadow = [p.clone().detach() for p in parameters]
        self.backup: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, parameters: list[nn.Parameter]):
        for s, p in zip(self.shadow, parameters):
            s.lerp_(p.data, 1.0 - self.decay)

    def apply_shadow(self, parameters: list[nn.Parameter]):
        """Replace model params with EMA params (for eval/sampling)."""
        self.backup = [p.data.clone() for p in parameters]
        for s, p in zip(self.shadow, parameters):
            p.data.copy_(s)

    def restore(self, parameters: list[nn.Parameter]):
        """Restore original model params after eval."""
        for b, p in zip(self.backup, parameters):
            p.data.copy_(b)
        self.backup = []

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict: dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class PoseConditionedSD(nn.Module):
    """Stable Diffusion with two-reference cross-attention conditioning.

    Target-anchored design: Plucker rays express reference poses relative to
    the target view. The UNet input is the standard 4-channel noisy latent
    (no pose concat), preserving the pretrained SD initialization.
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

        # Read cross-attention dim from UNet config instead of hardcoding
        cross_attn_dim = self.unet.config.cross_attention_dim
        self.ref_encoder = ReferenceEncoder(output_dim=cross_attn_dim).to(self.device)
        self._cache_null_embeddings()

    def configure_trainable(self, unet_train_mode: str = "full") -> None:
        if unet_train_mode == "full":
            self.unet.requires_grad_(True)
            return
        if unet_train_mode != "cond":
            raise ValueError(f"Unknown unet_train_mode: {unet_train_mode}")
        self.unet.requires_grad_(False)
        self.unet.conv_in.requires_grad_(True)
        # Enable gradients on individual parameters whose name contains "attn2"
        for name, param in self.unet.named_parameters():
            if "attn2" in name:
                param.requires_grad_(True)

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return all parameters that require gradients (UNet + ref_encoder)."""
        return (
            [p for p in self.unet.parameters() if p.requires_grad]
            + list(self.ref_encoder.parameters())
        )

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
        ref_drop_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build cross-attention context tensors.

        The ref encoder always sees real reference data.  Conditioning dropout
        is implemented by masking the *output* tokens to zero, so the
        unconditional signal seen during training exactly matches the
        ``torch.zeros_like(ref_tokens)`` used at inference time.

        Args:
            ref_drop_mask: (B,) bool tensor.  ``True`` = drop this sample's
                ref tokens (replace with zeros for unconditional training).
        """
        ref_tokens = self._encode_refs(ref1_latent, ref2_latent, plucker_ref1, plucker_ref2)

        # Training conditioning dropout: zero output tokens for dropped samples
        if ref_drop_mask is not None and ref_drop_mask.any():
            keep = (~ref_drop_mask).to(ref_tokens.dtype).view(-1, 1, 1)
            ref_tokens = ref_tokens * keep

        cond_context = torch.cat([text_cond, ref_tokens], dim=1)

        if text_uncond is None:
            return cond_context, None

        # Inference CFG: unconditional branch uses zero ref tokens
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
        ref_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond_context, _ = self.build_conditioning_contexts(
            ref1_latent, ref2_latent, plucker_ref1, plucker_ref2,
            text_cond=text_emb, ref_drop_mask=ref_drop_mask,
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

        # --- Conditioning dropout (joint text + ref) for CFG training ---
        # Text: replaced with null embedding.
        # Refs: encoder always sees real data; output tokens are zeroed via ref_drop_mask.
        # This ensures the unconditional signal matches inference exactly.
        ref_drop_mask = None
        if self.training and cond_drop_prob > 0:
            ref_drop_mask = torch.rand(B, device=self.device) < cond_drop_prob
            if ref_drop_mask.any():
                num_drop = int(ref_drop_mask.sum().item())
                text_emb = text_emb.clone()
                text_emb[ref_drop_mask] = self.null_text_emb.expand(num_drop, -1, -1)
            else:
                # No samples were dropped; pass None so we skip the mask logic
                ref_drop_mask = None

        noise_pred = self.forward(
            noisy_target, timesteps,
            ref1_latent, ref2_latent,
            plucker_ref1, plucker_ref2,
            text_emb,
            ref_drop_mask=ref_drop_mask,
        )
        loss = F.mse_loss(noise_pred, noise)
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


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_pose_sd_checkpoint(
    model: PoseConditionedSD,
    checkpoint_path: str | PathLike,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: object | None = None,
    ema: EMAModel | None = None,
    epoch: int = 0,
    global_step: int = 0,
) -> None:
    state: dict = {
        "unet": model.unet.state_dict(),
        "ref_encoder": model.ref_encoder.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None and hasattr(lr_scheduler, "state_dict"):
        state["lr_scheduler"] = lr_scheduler.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, checkpoint_path)


def load_pose_sd_checkpoint(
    model: PoseConditionedSD,
    checkpoint_path: str | PathLike,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: object | None = None,
    ema: EMAModel | None = None,
) -> dict:
    """Load checkpoint.  Returns ``{"epoch": int, "global_step": int}``."""
    state = torch.load(checkpoint_path, map_location=device)

    # Handle legacy checkpoints that are just a raw state_dict
    if not isinstance(state, dict) or "unet" not in state:
        model.unet.load_state_dict(state)
        return {"epoch": 0, "global_step": 0}

    model.unet.load_state_dict(state["unet"])
    model.ref_encoder.load_state_dict(state["ref_encoder"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if lr_scheduler is not None and "lr_scheduler" in state and hasattr(lr_scheduler, "load_state_dict"):
        lr_scheduler.load_state_dict(state["lr_scheduler"])
    if ema is not None and "ema" in state:
        ema.load_state_dict(state["ema"])

    return {
        "epoch": state.get("epoch", 0),
        "global_step": state.get("global_step", 0),
    }