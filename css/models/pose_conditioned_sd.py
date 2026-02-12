from os import PathLike

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm


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

    def forward(self, ref_latent, plucker_ref):
        x = torch.cat([ref_latent, plucker_ref], dim=1)
        return self.net(x).flatten(2).permute(0, 2, 1)


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
            new_conv_in.bias = old_conv_in.bias
            
        self.unet.conv_in = new_conv_in.to(device)
        self.ref_encoder = ReferenceEncoder().to(device)

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
            self.null_ref_tokens = torch.zeros(1, 512, 1024, device=self.device)

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

    def forward(self, noisy_target, timesteps, ref1_latent, ref2_latent, 
                plucker_ref1, plucker_ref2, plucker_target, text_emb):
        ref1_tokens = self.ref_encoder(ref1_latent, plucker_ref1)
        ref2_tokens = self.ref_encoder(ref2_latent, plucker_ref2)
        cond_context = torch.cat([text_emb, ref1_tokens, ref2_tokens], dim=1)

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

        if self.training and torch.rand(1).item() < cond_drop_prob:
            text_emb = self.null_text_emb.expand(B, -1, -1)
            ref1_latent = torch.zeros_like(ref1_latent)
            ref2_latent = torch.zeros_like(ref2_latent)
            plucker_ref1 = torch.zeros_like(plucker_ref1)
            plucker_ref2 = torch.zeros_like(plucker_ref2)

        loss = F.mse_loss(
            self.forward(noisy_target, timesteps, ref1_latent, ref2_latent,
                         plucker_ref1, plucker_ref2, plucker_target, text_emb),
            noise
        )
        return loss

    @torch.inference_mode()
    def sample(self, ref1_img, ref2_img, plucker_ref1, plucker_ref2, plucker_target, 
               prompt="", num_steps=50, cfg_scale=1.0, target=None, start_t=1000):
        
        B = ref1_img.shape[0]
        
        ref1_latent = self.encode_image(ref1_img.to(self.device))
        ref2_latent = self.encode_image(ref2_img.to(self.device))
        
        plucker_ref1 = plucker_ref1.to(self.device)
        plucker_ref2 = plucker_ref2.to(self.device)
        plucker_target = plucker_target.to(self.device)

        ref1_tokens = self.ref_encoder(ref1_latent, plucker_ref1)
        ref2_tokens = self.ref_encoder(ref2_latent, plucker_ref2)
        ref_tokens = torch.cat([ref1_tokens, ref2_tokens], dim=1)

        text_tokens = self.tokenizer([prompt], padding="max_length", max_length=77, return_tensors="pt")
        text_cond = self.text_encoder(text_tokens.input_ids.to(self.device))[0].expand(B, -1, -1)
        cond_context = torch.cat([text_cond, ref_tokens], dim=1)

        null_text_batch = self.null_text_emb.expand(B, -1, -1)
        null_ref_batch = self.null_ref_tokens.expand(B, -1, -1)
        uncond_context = torch.cat([null_text_batch, null_ref_batch], dim=1)

        self.scheduler.set_timesteps(num_steps)
        
        if target is not None:
            target_latent = self.encode_image(target.to(self.device))
            noise = torch.randn_like(target_latent)
            start_t = int(start_t)
            max_t = int(self.scheduler.config.num_train_timesteps) - 1
            start_t = max(0, min(max_t, start_t))
            timestep = torch.tensor([start_t], device=self.device, dtype=torch.long)
            latent = self.scheduler.add_noise(target_latent, noise, timestep)
            timesteps = self.scheduler.timesteps[self.scheduler.timesteps <= start_t]
        else:
            latent = torch.randn_like(ref1_latent)
            timesteps = self.scheduler.timesteps
    
        for t in tqdm(timesteps, desc="Sampling"):
            unet_input = torch.cat([latent, plucker_target], dim=1)
            eps_uncond = self.unet(unet_input, t, encoder_hidden_states=uncond_context).sample
            eps_cond = self.unet(unet_input, t, encoder_hidden_states=cond_context).sample
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
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
