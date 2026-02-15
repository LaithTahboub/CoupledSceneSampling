from os import PathLike

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from css.models.apg import AdaptiveProjectedGuidance


class ReferenceEncoder(nn.Module):
    """Encode (reference latent + Plucker) into cross-attention tokens."""

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
            nn.Conv2d(hidden_dim, output_dim, 1) 
        )

    def encode_map(self, ref_latent, plucker_ref):
        x = torch.cat([ref_latent, plucker_ref], dim=1)
        return self.net(x)

    def forward(self, ref_latent, plucker_ref):
        return self.encode_map(ref_latent, plucker_ref).flatten(2).permute(0, 2, 1)


class PoseMapEncoder(nn.Module):
    """Encode target Plucker map into cross-attention tokens."""

    def __init__(self, plucker_dim=6, hidden_dim=768, output_dim=1024):
        super().__init__()
        final = nn.Conv2d(hidden_dim, output_dim, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

        self.net = nn.Sequential(
            nn.Conv2d(plucker_dim, 256, 3, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            final,
        )

    def encode_map(self, plucker_target):
        return self.net(plucker_target)

    def forward(self, plucker_target):
        return self.encode_map(plucker_target).flatten(2).permute(0, 2, 1)


class EpipolarCrossViewAttention(nn.Module):
    """Cross-attention with epipolar priors from Plucker rays."""

    def __init__(self, channels=1024, attn_dim=256, top_k=32, geo_temperature=0.1, eps=1e-6):
        super().__init__()
        self.top_k = int(top_k)
        self.geo_temperature = float(geo_temperature)
        self.eps = float(eps)
        self.scale = attn_dim**-0.5

        self.norm_q = nn.LayerNorm(channels)
        self.norm_k = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, attn_dim)
        self.k_proj = nn.Linear(channels, attn_dim)
        self.v_proj = nn.Linear(channels, attn_dim)
        self.out_proj = nn.Linear(attn_dim, channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _flatten_map(self, feat: torch.Tensor) -> torch.Tensor:
        return feat.flatten(2).permute(0, 2, 1)

    def _flatten_rays(self, plucker: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        rays = F.interpolate(plucker, size=(h, w), mode="bilinear", align_corners=False)
        rays = rays.permute(0, 2, 3, 1).reshape(rays.shape[0], h * w, 6)
        # SEVA convention: [ray_direction(3), moment(3)].
        dirs = F.normalize(rays[..., :3], dim=-1, eps=self.eps)
        moments = rays[..., 3:]
        return moments, dirs

    def _epipolar_bias(
        self,
        plucker_query: torch.Tensor,
        plucker_key: torch.Tensor,
        h: int,
        w: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        m_q, d_q = self._flatten_rays(plucker_query, h, w)
        m_k, d_k = self._flatten_rays(plucker_key, h, w)

        reciprocal = torch.abs(
            (d_q.unsqueeze(2) * m_k.unsqueeze(1)).sum(dim=-1)
            + (d_k.unsqueeze(1) * m_q.unsqueeze(2)).sum(dim=-1)
        )
        reciprocal = reciprocal / (m_q.norm(dim=-1, keepdim=True) + m_k.norm(dim=-1).unsqueeze(1) + self.eps)

        geo_bias = -reciprocal / self.geo_temperature
        if self.top_k > 0 and self.top_k < geo_bias.shape[-1]:
            keep = reciprocal.topk(self.top_k, dim=-1, largest=False).indices
            drop_mask = torch.ones_like(geo_bias, dtype=torch.bool)
            drop_mask.scatter_(2, keep, False)
            geo_bias = geo_bias.masked_fill(drop_mask, -1e4)

        return geo_bias.to(device=device, dtype=dtype)

    def forward(
        self,
        query_map: torch.Tensor,
        key_value_map: torch.Tensor,
        plucker_query: torch.Tensor,
        plucker_key: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = query_map.shape

        q_tokens = self._flatten_map(query_map)
        kv_tokens = self._flatten_map(key_value_map)
        q = self.q_proj(self.norm_q(q_tokens))
        k = self.k_proj(self.norm_k(kv_tokens))
        v = self.v_proj(kv_tokens)

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        logits = logits + self._epipolar_bias(
            plucker_query, plucker_key, h, w, logits.dtype, logits.device
        )

        attn = torch.softmax(logits, dim=-1)
        fused = torch.matmul(attn, v)
        fused = self.out_proj(fused)
        return fused.permute(0, 2, 1).reshape(b, query_map.shape[1], h, w)


class PoseConditionedSD(nn.Module):
    """Stable Diffusion with pose concat and two-reference cross-attention."""

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

        old_conv_in = self.unet.conv_in
        new_conv_in = nn.Conv2d(
            10, old_conv_in.out_channels, 
            kernel_size=old_conv_in.kernel_size, 
            padding=old_conv_in.padding
        )
        
        with torch.no_grad():
            new_conv_in.weight[:, :4] = old_conv_in.weight
            new_conv_in.weight[:, 4:] = 0
            new_conv_in.bias.copy_(old_conv_in.bias)
            
        self.unet.conv_in = new_conv_in.to(device)
        self.ref_encoder = ReferenceEncoder().to(device)
        self.target_pose_encoder = PoseMapEncoder().to(device)
        self.epipolar_attn = EpipolarCrossViewAttention().to(device)

        self._cache_null_embeddings()

    def configure_trainable(self, unet_train_mode: str = "full") -> None:
        """Configure which UNet weights are trainable."""
        if unet_train_mode == "full":
            self.unet.requires_grad_(True)
            return

        if unet_train_mode != "cond":
            raise ValueError(f"Unknown unet_train_mode: {unet_train_mode}")

        # Keep UNet close to pretrained prior and focus updates on conditioning paths.
        self.unet.requires_grad_(False)
        self.unet.conv_in.requires_grad_(True)
        for name, module in self.unet.named_modules():
            if "attn2" in name:
                module.requires_grad_(True)

    def _cache_null_embeddings(self):
        with torch.no_grad():
            tokens = self.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            self.null_text_emb = self.text_encoder(tokens.input_ids.to(self.device))[0]

    @torch.no_grad()
    def get_text_embedding(self, prompt: str) -> torch.Tensor:
        if prompt == "":
            return self.null_text_emb
        if not hasattr(self, '_cached_prompt') or self._cached_prompt != prompt:
            tokens = self.tokenizer([prompt], padding="max_length", max_length=77, return_tensors="pt")
            self._cached_text_emb = self.text_encoder(tokens.input_ids.to(self.device))[0]
            self._cached_prompt = prompt
        return self._cached_text_emb

    @torch.no_grad()
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample

    @staticmethod
    def _x0_from_eps(latent: torch.Tensor, eps: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = alpha_bar.sqrt()
        sqrt_one_minus = (1.0 - alpha_bar).clamp_min(0.0).sqrt()
        return (latent - sqrt_one_minus * eps) / sqrt_alpha

    @staticmethod
    def _eps_from_x0(latent: torch.Tensor, x0: torch.Tensor, alpha_bar: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        sqrt_alpha = alpha_bar.sqrt()
        sqrt_one_minus = (1.0 - alpha_bar).clamp_min(0.0).sqrt().clamp_min(eps)
        return (latent - sqrt_alpha * x0) / sqrt_one_minus

    def _build_pose_and_ref_tokens(
        self,
        ref1_latent: torch.Tensor,
        ref2_latent: torch.Tensor,
        plucker_ref1: torch.Tensor,
        plucker_ref2: torch.Tensor,
        plucker_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_map = self.target_pose_encoder.encode_map(plucker_target)
        target_tokens = target_map.flatten(2).permute(0, 2, 1)

        ref1_map = self.ref_encoder.encode_map(ref1_latent, plucker_ref1)
        ref2_map = self.ref_encoder.encode_map(ref2_latent, plucker_ref2)

        ref1_epi = self.epipolar_attn(target_map, ref1_map, plucker_target, plucker_ref1)
        ref2_epi = self.epipolar_attn(target_map, ref2_map, plucker_target, plucker_ref2)

        ref1_tokens = (ref1_map + ref1_epi).flatten(2).permute(0, 2, 1)
        ref2_tokens = (ref2_map + ref2_epi).flatten(2).permute(0, 2, 1)
        ref_tokens = torch.cat([ref1_tokens, ref2_tokens], dim=1)
        return target_tokens, ref_tokens

    def build_conditioning_contexts(
        self,
        ref1_latent: torch.Tensor,
        ref2_latent: torch.Tensor,
        plucker_ref1: torch.Tensor,
        plucker_ref2: torch.Tensor,
        plucker_target: torch.Tensor,
        text_cond: torch.Tensor,
        text_uncond: torch.Tensor | None = None,
        ref_keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        target_tokens, ref_tokens = self._build_pose_and_ref_tokens(
            ref1_latent, ref2_latent, plucker_ref1, plucker_ref2, plucker_target
        )
        if ref_keep_mask is not None:
            keep = ref_keep_mask.to(ref_tokens.dtype).view(ref_tokens.shape[0], 1, 1)
            ref_tokens = ref_tokens * keep
        cond_context = torch.cat([text_cond, target_tokens, ref_tokens], dim=1)
        if text_uncond is None:
            return cond_context, None

        uncond_context = torch.cat([text_uncond, target_tokens, torch.zeros_like(ref_tokens)], dim=1)
        return cond_context, uncond_context

    def forward(self, noisy_target, timesteps, ref1_latent, ref2_latent, 
                plucker_ref1, plucker_ref2, plucker_target, text_emb, ref_keep_mask=None):
        cond_context, _ = self.build_conditioning_contexts(
            ref1_latent,
            ref2_latent,
            plucker_ref1,
            plucker_ref2,
            plucker_target,
            text_cond=text_emb,
            ref_keep_mask=ref_keep_mask,
        )

        unet_input = torch.cat([noisy_target, plucker_target], dim=1)
        return self.unet(unet_input, timesteps, encoder_hidden_states=cond_context).sample

    def training_step(
        self,
        batch: dict,
        prompt: str = "",
        cond_drop_prob: float = 0.1,
        min_timestep: int = 0,
        max_timestep: int | None = None,
    ) -> torch.Tensor:
        ref1_img = batch["ref1_img"].to(self.device)
        ref2_img = batch["ref2_img"].to(self.device)
        target_img = batch["target_img"].to(self.device)

        plucker_ref1 = batch["plucker_ref1"].to(self.device)
        plucker_ref2 = batch["plucker_ref2"].to(self.device)
        plucker_target = batch["plucker_target"].to(self.device)

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

        text_emb = self.get_text_embedding(prompt).expand(B, -1, -1)

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
        else:
            drop_mask = None

        loss = F.mse_loss(
            self.forward(noisy_target, timesteps, ref1_latent, ref2_latent,
                         plucker_ref1, plucker_ref2, plucker_target, text_emb,
                         ref_keep_mask=(None if drop_mask is None else ~drop_mask)),
            noise
        )
        return loss

    @torch.inference_mode()
    def sample(self, ref1_img, ref2_img, plucker_ref1, plucker_ref2, plucker_target, 
               prompt="", num_steps=50, cfg_scale=1.0, target=None, start_t=1000,
               apg_eta=0.0, apg_momentum=-0.5, apg_norm_threshold=0.0):
        
        B = ref1_img.shape[0]
        
        ref1_latent = self.encode_image(ref1_img.to(self.device))
        ref2_latent = self.encode_image(ref2_img.to(self.device))
        
        plucker_ref1 = plucker_ref1.to(self.device)
        plucker_ref2 = plucker_ref2.to(self.device)
        plucker_target = plucker_target.to(self.device)

        text_cond = self.get_text_embedding(prompt).expand(B, -1, -1)
        null_text_batch = self.null_text_emb.expand(B, -1, -1)
        cond_context, uncond_context = self.build_conditioning_contexts(
            ref1_latent,
            ref2_latent,
            plucker_ref1,
            plucker_ref2,
            plucker_target,
            text_cond=text_cond,
            text_uncond=null_text_batch,
        )

        self.scheduler.set_timesteps(num_steps)
        
        if target is not None:
            target_latent = self.encode_image(target.to(self.device))
            noise = torch.randn_like(target_latent)
            start_t = int(start_t)
            max_t = int(self.scheduler.config.num_train_timesteps) - 1
            start_t = max(0, min(max_t, start_t))
            timestep = torch.full((B,), start_t, device=self.device, dtype=torch.long)
            latent = self.scheduler.add_noise(target_latent, noise, timestep)
            timesteps = self.scheduler.timesteps[self.scheduler.timesteps <= start_t]
        else:
            latent = torch.randn_like(ref1_latent)
            timesteps = self.scheduler.timesteps

        guider = AdaptiveProjectedGuidance(
            guidance_scale=cfg_scale,
            eta=apg_eta,
            momentum=apg_momentum,
            norm_threshold=apg_norm_threshold,
        )
        guider.reset()

        for t in tqdm(timesteps, desc="Sampling"):
            t_val = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            unet_input = torch.cat([latent, plucker_target], dim=1)
            unet_pair = torch.cat([unet_input, unet_input], dim=0)
            context_pair = torch.cat([uncond_context, cond_context], dim=0)
            t_pair = torch.full((2 * B,), t_val, device=self.device, dtype=torch.long)
            eps_pair = self.unet(unet_pair, t_pair, encoder_hidden_states=context_pair).sample
            eps_uncond, eps_cond = eps_pair.chunk(2)
            alpha_bar = self.scheduler.alphas_cumprod[t_val].to(device=latent.device, dtype=latent.dtype)
            if float((1.0 - alpha_bar).item()) <= 1e-8:
                eps = eps_cond
            else:
                x0_uncond = self._x0_from_eps(latent, eps_uncond, alpha_bar)
                x0_cond = self._x0_from_eps(latent, eps_cond, alpha_bar)
                x0_guided = guider.guide(x0_cond, x0_uncond)
                eps = self._eps_from_x0(latent, x0_guided, alpha_bar)
            latent = self.scheduler.step(eps, t, latent).prev_sample

        return self.decode_latent(latent)


def save_pose_sd_checkpoint(model: PoseConditionedSD, checkpoint_path: str | PathLike) -> None:
    torch.save(
        {
            "unet": model.unet.state_dict(),
            "ref_encoder": model.ref_encoder.state_dict(),
            "target_pose_encoder": model.target_pose_encoder.state_dict(),
            "epipolar_attn": model.epipolar_attn.state_dict(),
        },
        checkpoint_path,
    )


def load_pose_sd_checkpoint(model: PoseConditionedSD, checkpoint_path: str | PathLike, device: str) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "unet" in state:
        model.unet.load_state_dict(state["unet"])
        model.ref_encoder.load_state_dict(state["ref_encoder"])
        if "target_pose_encoder" in state:
            model.target_pose_encoder.load_state_dict(state["target_pose_encoder"])
        else:
            print("[Warning] Checkpoint missing target_pose_encoder; using fresh initialization.")
        if "epipolar_attn" in state:
            model.epipolar_attn.load_state_dict(state["epipolar_attn"])
        else:
            print("[Warning] Checkpoint missing epipolar_attn; using fresh initialization.")
    else:
        model.unet.load_state_dict(state)
