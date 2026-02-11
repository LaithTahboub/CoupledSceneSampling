import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm


class ReferenceEncoder(nn.Module):
    """
    Encode reference frames + Pluckers to visual tokens for cross-attention.
    Now preserves spatial resolution better (16x16 grid) instead of pooling to 8x8.
    """

    def __init__(self, latent_dim=4, plucker_dim=6, hidden_dim=768, output_dim=1024):
        super().__init__()
        # Input: Ref Latent (4) + Ref Plucker (6) = 10 channels
        in_channels = latent_dim + plucker_dim 

        self.net = nn.Sequential(
            # Block 1: 10 -> 256
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            
            # Downsample 1: 64x64 -> 32x32
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.SiLU(),

            # Downsample 2: 32x32 -> 16x16
            nn.Conv2d(512, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            
            # Final projection to cross-attention dim (usually 1024 for SD 2.1)
            nn.Conv2d(hidden_dim, output_dim, 1) 
        )

    def forward(self, ref_latent, plucker_ref):
        """
        Args:
            ref_latent: (B, 4, 64, 64)
            plucker_ref: (B, 6, 64, 64)
        Returns: 
            (B, 256, 1024) - 256 spatial tokens per reference
        """
        x = torch.cat([ref_latent, plucker_ref], dim=1)
        feat = self.net(x)  # (B, 1024, 16, 16)
        
        # Flatten spatial dims to create sequence of tokens
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        tokens = feat.flatten(2).permute(0, 2, 1) 
        return tokens


class PoseConditionedSD(nn.Module):
    """
    SD with:
    1. Target Pose Concatenation (CAT3D style) -> Input to UNet
    2. Cross-Frame Attention -> References via Cross-Attn
    """

    def __init__(self, pretrained_model: str = "manojb/stable-diffusion-2-1-base", device: str = "cuda"):
        super().__init__()
        self.device = device

        # Load pretrained SD components
        self.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet").to(device)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # ---------------------------------------------------------
        # ARCHITECTURE CHANGE 1: Modify UNet input for concatenation
        # Standard SD input is 4 channels. We need 4 (latent) + 6 (plucker) = 10.
        # ---------------------------------------------------------
        old_conv_in = self.unet.conv_in
        new_conv_in = nn.Conv2d(
            10, old_conv_in.out_channels, 
            kernel_size=old_conv_in.kernel_size, 
            padding=old_conv_in.padding
        )
        
        # Initialize new weights:
        # Copy original 4 channels to preserve pretrained knowledge
        with torch.no_grad():
            new_conv_in.weight[:, :4] = old_conv_in.weight
            # Initialize pose channels to zero (Zero-Conv strategy)
            # This ensures training starts as if standard SD, slowly learning pose
            new_conv_in.weight[:, 4:] = 0
            new_conv_in.bias = old_conv_in.bias
            
        self.unet.conv_in = new_conv_in.to(device)

        # ---------------------------------------------------------
        # ARCHITECTURE CHANGE 2: Reference Encoder
        # ---------------------------------------------------------
        self.ref_encoder = ReferenceEncoder().to(device)

        self._cache_null_embeddings()

    def _cache_null_embeddings(self):
        """Cache null text and reference embeddings for CFG."""
        with torch.no_grad():
            tokens = self.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            self.null_text_emb = self.text_encoder(tokens.input_ids.to(self.device))[0]
            
            # Null ref tokens: shape (1, 256*2, 1024) -> (1, 512, 1024)
            # We assume 2 references max for now
            self.null_ref_tokens = torch.zeros(1, 512, 1024, device=self.device)

    @torch.no_grad()
    def get_text_embedding(self, prompt: str) -> torch.Tensor:
        """Get text embedding, cached for repeated prompts."""
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
        
        # 1. Encode References (Appearance + Pose) -> Cross-Attention Tokens
        ref1_tokens = self.ref_encoder(ref1_latent, plucker_ref1) # (B, 256, 1024)
        ref2_tokens = self.ref_encoder(ref2_latent, plucker_ref2) # (B, 256, 1024)
        
        # Concatenate text + visual references
        # (B, 77, 1024) + (B, 512, 1024) = (B, 589, 1024)
        cond_context = torch.cat([text_emb, ref1_tokens, ref2_tokens], dim=1)

        # 2. Prepare UNet Input (Target Latent + Target Pose)
        # Concatenate along channel dimension: (B, 4, H, W) + (B, 6, H, W) -> (B, 10, H, W)
        unet_input = torch.cat([noisy_target, plucker_target], dim=1)

        # 3. Predict noise
        return self.unet(unet_input, timesteps, encoder_hidden_states=cond_context).sample

    def training_step(self, batch: dict, prompt: str = "", cond_drop_prob: float = 0.1) -> torch.Tensor:
        ref1_img = batch["ref1_img"].to(self.device)
        ref2_img = batch["ref2_img"].to(self.device)
        target_img = batch["target_img"].to(self.device)

        plucker_ref1 = batch["plucker_ref1"].to(self.device)
        plucker_ref2 = batch["plucker_ref2"].to(self.device)
        plucker_target = batch["plucker_target"].to(self.device)

        B = target_img.shape[0]

        # VAE Encoding
        ref1_latent = self.encode_image(ref1_img)
        ref2_latent = self.encode_image(ref2_img)
        target_latent = self.encode_image(target_img)

        # Noise Addition
        noise = torch.randn_like(target_latent)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=self.device)
        noisy_target = self.scheduler.add_noise(target_latent, noise, timesteps)

        # Text Embeddings (cached)
        text_emb = self.get_text_embedding(prompt).expand(B, -1, -1)

        # Conditioning dropout for CFG training: with probability cond_drop_prob,
        # replace text + ref conditioning with null so the model learns a
        # meaningful unconditional baseline for classifier-free guidance.
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
               prompt="", num_steps=50, cfg_scale=2.0):
        
        B = ref1_img.shape[0] # Handle batch size > 1
        
        ref1_latent = self.encode_image(ref1_img.to(self.device))
        ref2_latent = self.encode_image(ref2_img.to(self.device))
        
        # Ensure Pluckers are on device
        plucker_ref1 = plucker_ref1.to(self.device)
        plucker_ref2 = plucker_ref2.to(self.device)
        plucker_target = plucker_target.to(self.device)

        # Encode References
        ref1_tokens = self.ref_encoder(ref1_latent, plucker_ref1)
        ref2_tokens = self.ref_encoder(ref2_latent, plucker_ref2)
        ref_tokens = torch.cat([ref1_tokens, ref2_tokens], dim=1) # (B, 512, 1024)

        # Text Embeddings
        text_tokens = self.tokenizer([prompt], padding="max_length", max_length=77, return_tensors="pt")
        text_cond = self.text_encoder(text_tokens.input_ids.to(self.device))[0].expand(B, -1, -1)
        
        # Classifier-Free Guidance Setup
        # Conditional Context
        cond_context = torch.cat([text_cond, ref_tokens], dim=1)
        
        # Unconditional Context (Null text + Null refs)
        null_text_batch = self.null_text_emb.expand(B, -1, -1)
        null_ref_batch = self.null_ref_tokens.expand(B, -1, -1)
        uncond_context = torch.cat([null_text_batch, null_ref_batch], dim=1)
        
        # Initial Noise
        latent = torch.randn_like(ref1_latent)
        self.scheduler.set_timesteps(num_steps)

        for t in tqdm(self.scheduler.timesteps, desc="Sampling"):
            # Double the batch for CFG to avoid two forward passes? 
            # Usually better to do two passes if memory is tight, or concat batch.
            # Here we do standard two-pass for clarity.
            
            # 1. Unconditional Pass
            # Note: We MUST concatenate plucker_target here too! 
            # The geometry is NOT dropped in CFG, only the conditioning (text/ref appearance).
            unet_input_uncond = torch.cat([latent, plucker_target], dim=1)
            eps_uncond = self.unet(unet_input_uncond, t, encoder_hidden_states=uncond_context).sample

            # 2. Conditional Pass
            unet_input_cond = torch.cat([latent, plucker_target], dim=1)
            eps_cond = self.unet(unet_input_cond, t, encoder_hidden_states=cond_context).sample

            # Combine
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            latent = self.scheduler.step(eps, t, latent).prev_sample

        return self.decode_latent(latent)
