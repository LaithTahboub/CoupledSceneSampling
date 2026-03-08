"""
Pose-Conditioned Stable Diffusion

Conditions SD on two reference images + their poses + target pose.
Two references reduce geometric ambiguity (like stereo vision).

Input channels (concatenated):
  - 4: noisy target latent
  - 4: reference image 1 latent
  - 4: reference image 2 latent
  - 6: reference pose 1 (Plucker)
  - 6: reference pose 2 (Plucker)
  - 6: target pose (Plucker)
  Total: 30 channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from seva.geometry import get_plucker_coordinates, to_hom_pose, get_default_intrinsics


class MegaScenesDataset(Dataset):
    """
    Loads triplets from MegaScenes: (ref1, ref2, target) with poses.

    Expected directory structure:
        scene_dir/
            images/
                img_0000.jpg, img_0001.jpg, ...
            poses/
                pose_0000.npy, pose_0001.npy, ...  (4x4 c2w matrices)
            intrinsics.npy  (3x3 K matrix or single fov value)

    Training is always in "generation" mode (target between refs).
    At inference, choose mode via plucker_target:
        - Generation: plucker_target = actual novel pose (NVS)
        - Reconstruction: plucker_target = nearest ref pose (appearance transfer)
    """

    def __init__(self, scene_dirs: list[str], H: int = 512, W: int = 512):
        self.H, self.W = H, W
        self.latent_h, self.latent_w = H // 8, W // 8

        self.samples = []  # List of (scene_dir, indices) for valid triplets

        for scene_dir in scene_dirs:
            scene_path = Path(scene_dir)
            image_files = sorted(scene_path.glob("images/*.jpg")) + sorted(scene_path.glob("images/*.png"))
            n_images = len(image_files)

            if n_images < 3:
                continue

            # Load all poses for this scene
            poses = []
            for i in range(n_images):
                pose_file = scene_path / "poses" / f"pose_{i:04d}.npy"
                if pose_file.exists():
                    poses.append(np.load(pose_file))
                else:
                    break

            if len(poses) < 3:
                continue

            # Triplets where target is between refs (novel view synthesis)
            for i in range(n_images):
                for j in range(i + 2, min(i + 10, n_images)):
                    for k in range(i + 1, j):
                        self.samples.append((scene_path, [i, j, k]))

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.W, self.H))
        arr = np.array(img) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).float() * 2 - 1

    def _compute_plucker(self, c2w: np.ndarray, K: np.ndarray) -> torch.Tensor:
        """Compute Plucker coordinates for a single view relative to itself (identity transform)."""
        c2w = torch.from_numpy(c2w).float()
        K = torch.from_numpy(K).float()

        c2w = to_hom_pose(c2w.unsqueeze(0))
        w2c = torch.linalg.inv(c2w)

        # Plucker coords relative to self
        plucker = get_plucker_coordinates(
            extrinsics_src=w2c[0],
            extrinsics=w2c,
            intrinsics=K.unsqueeze(0),
            target_size=[self.latent_h, self.latent_w]
        )
        return plucker[0]  # (6, h, w)

    def _compute_relative_plucker(self, c2w_src: np.ndarray, c2w_tgt: np.ndarray, K: np.ndarray) -> torch.Tensor:
        """Compute Plucker coordinates for target view relative to source view."""
        c2w_src = torch.from_numpy(c2w_src).float().unsqueeze(0)
        c2w_tgt = torch.from_numpy(c2w_tgt).float().unsqueeze(0)
        K = torch.from_numpy(K).float().unsqueeze(0)

        c2w_src = to_hom_pose(c2w_src)
        c2w_tgt = to_hom_pose(c2w_tgt)
        w2c_src = torch.linalg.inv(c2w_src)
        w2c_tgt = torch.linalg.inv(c2w_tgt)

        plucker = get_plucker_coordinates(
            extrinsics_src=w2c_src[0],
            extrinsics=w2c_tgt,
            intrinsics=K,
            target_size=[self.latent_h, self.latent_w]
        )
        return plucker[0]  # (6, h, w)

    def __getitem__(self, idx):
        scene_path, (i, j, k) = self.samples[idx]

        # Load images
        image_files = sorted(scene_path.glob("images/*.jpg")) + sorted(scene_path.glob("images/*.png"))
        ref1_img = self._load_image(image_files[i])
        ref2_img = self._load_image(image_files[j])
        target_img = self._load_image(image_files[k])

        # Load poses
        pose_ref1 = np.load(scene_path / "poses" / f"pose_{i:04d}.npy")
        pose_ref2 = np.load(scene_path / "poses" / f"pose_{j:04d}.npy")
        pose_target = np.load(scene_path / "poses" / f"pose_{k:04d}.npy")

        # Load intrinsics (assume same for all images in scene)
        intrinsics_file = scene_path / "intrinsics.npy"
        if intrinsics_file.exists():
            K = np.load(intrinsics_file)
            if K.shape == ():  # Single fov value
                fov = float(K)
                K = np.array([
                    [self.W / (2 * np.tan(fov / 2)), 0, self.W / 2],
                    [0, self.H / (2 * np.tan(fov / 2)), self.H / 2],
                    [0, 0, 1]
                ])
        else:
            # Default intrinsics (45 degree fov)
            fov = np.pi / 4
            K = np.array([
                [self.W / (2 * np.tan(fov / 2)), 0, self.W / 2],
                [0, self.H / (2 * np.tan(fov / 2)), self.H / 2],
                [0, 0, 1]
            ])

        # Compute Plucker coordinates (all relative to target view)
        plucker_ref1 = self._compute_relative_plucker(pose_target, pose_ref1, K)
        plucker_ref2 = self._compute_relative_plucker(pose_target, pose_ref2, K)
        plucker_target = self._compute_relative_plucker(pose_target, pose_target, K)  # Identity

        return {
            "ref1_img": ref1_img,
            "ref2_img": ref2_img,
            "target_img": target_img,
            "plucker_ref1": plucker_ref1,
            "plucker_ref2": plucker_ref2,
            "plucker_target": plucker_target,
        }


class PoseConditionedSD(nn.Module):
    """
    Stable Diffusion conditioned on two reference images and poses.

    Modifies the UNet's input conv to accept 30 channels:
    [noisy_target (4) | ref1_latent (4) | ref2_latent (4) |
     plucker_ref1 (6) | plucker_ref2 (6) | plucker_target (6)]
    """

    def __init__(self, pretrained_model: str = "stabilityai/stable-diffusion-2-1-base", device: str = "cuda"):
        super().__init__()
        self.device = device

        # Load pretrained components
        self.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet").to(device)
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
        self.scheduler = DDPMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")

        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Modify UNet input conv: 4 -> 30 channels
        self._expand_input_conv(in_channels=30)

        # Null text embedding for unconditional generation
        self._cache_null_text_emb()

    def _expand_input_conv(self, in_channels: int):
        """Expand UNet's input conv from 4 to 30 channels, preserving pretrained weights."""
        old_conv = self.unet.conv_in

        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        ).to(self.device)

        # Initialize: copy old weights for first 4 channels, zero for rest
        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, :4] = old_conv.weight
            new_conv.bias.copy_(old_conv.bias)

        self.unet.conv_in = new_conv

    def _cache_null_text_emb(self):
        """Cache null text embedding for classifier-free guidance."""
        with torch.no_grad():
            tokens = self.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
            self.null_text_emb = self.text_encoder(tokens.input_ids.to(self.device))[0]

    @torch.no_grad()
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space. img: (B, 3, H, W) in [-1, 1]."""
        latent = self.vae.encode(img).latent_dist.sample()
        return latent * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image. Returns (B, 3, H, W) in [-1, 1]."""
        latent = latent / self.vae.config.scaling_factor
        return self.vae.decode(latent).sample

    def get_text_embedding(self, prompt: str) -> torch.Tensor:
        """Get CLIP text embedding for prompt."""
        tokens = self.tokenizer([prompt], padding="max_length", max_length=77, return_tensors="pt")
        return self.text_encoder(tokens.input_ids.to(self.device))[0]

    def forward(
        self,
        noisy_target: torch.Tensor,  # (B, 4, h, w)
        timesteps: torch.Tensor,      # (B,)
        ref1_latent: torch.Tensor,    # (B, 4, h, w)
        ref2_latent: torch.Tensor,    # (B, 4, h, w)
        plucker_ref1: torch.Tensor,   # (B, 6, h, w)
        plucker_ref2: torch.Tensor,   # (B, 6, h, w)
        plucker_target: torch.Tensor, # (B, 6, h, w)
        text_emb: torch.Tensor,       # (B, 77, 1024)
    ) -> torch.Tensor:
        """
        Forward pass: predict noise given all conditioning.

        Returns: predicted noise (B, 4, h, w)
        """
        # Concatenate all inputs along channel dimension
        model_input = torch.cat([
            noisy_target,
            ref1_latent,
            ref2_latent,
            plucker_ref1,
            plucker_ref2,
            plucker_target,
        ], dim=1)  # (B, 30, h, w)

        # UNet forward
        noise_pred = self.unet(model_input, timesteps, encoder_hidden_states=text_emb).sample

        return noise_pred

    def training_step(self, batch: dict, prompt: str = "") -> torch.Tensor:
        """
        Single training step. Returns loss.

        batch contains: ref1_img, ref2_img, target_img, plucker_ref1, plucker_ref2, plucker_target
        """
        ref1_img = batch["ref1_img"].to(self.device)
        ref2_img = batch["ref2_img"].to(self.device)
        target_img = batch["target_img"].to(self.device)
        plucker_ref1 = batch["plucker_ref1"].to(self.device)
        plucker_ref2 = batch["plucker_ref2"].to(self.device)
        plucker_target = batch["plucker_target"].to(self.device)

        B = target_img.shape[0]

        # Encode images to latents
        with torch.no_grad():
            ref1_latent = self.encode_image(ref1_img)
            ref2_latent = self.encode_image(ref2_img)
            target_latent = self.encode_image(target_img)

        # Sample noise and timesteps
        noise = torch.randn_like(target_latent)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=self.device)

        # Add noise to target
        noisy_target = self.scheduler.add_noise(target_latent, noise, timesteps)

        # Text embedding (use empty prompt or scene description)
        if prompt:
            text_emb = self.get_text_embedding(prompt)
            text_emb = text_emb.expand(B, -1, -1)
        else:
            text_emb = self.null_text_emb.expand(B, -1, -1)

        # Predict noise
        noise_pred = self.forward(
            noisy_target, timesteps,
            ref1_latent, ref2_latent,
            plucker_ref1, plucker_ref2, plucker_target,
            text_emb
        )

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.inference_mode()
    def sample(
        self,
        ref1_img: torch.Tensor,       # (1, 3, H, W) in [-1, 1]
        ref2_img: torch.Tensor,       # (1, 3, H, W) in [-1, 1]
        plucker_ref1: torch.Tensor,   # (1, 6, h, w)
        plucker_ref2: torch.Tensor,   # (1, 6, h, w)
        plucker_target: torch.Tensor, # (1, 6, h, w)
        prompt: str = "",
        num_steps: int = 50,
        cfg_scale: float = 7.5,
        mode: str = "generation",     # "generation" or "reconstruction"
    ) -> torch.Tensor:
        """
        Sample a new view given two references and target pose.

        Args:
            mode: Controls what plucker_target represents:
                - "generation": plucker_target is the actual novel view pose.
                  Model synthesizes new geometry. Use for NVS with sparse inputs.
                - "reconstruction": plucker_target should be set to nearest ref's pose.
                  Model just transfers appearance (relighting, remove transients).
                  Use when you have many inputs and want to harmonize them.

        Returns: generated image (1, 3, H, W) in [-1, 1]
        """
        # Encode reference images
        ref1_latent = self.encode_image(ref1_img.to(self.device))
        ref2_latent = self.encode_image(ref2_img.to(self.device))

        # Move pluckers to device
        plucker_ref1 = plucker_ref1.to(self.device)
        plucker_ref2 = plucker_ref2.to(self.device)
        plucker_target = plucker_target.to(self.device)

        # Text embeddings for CFG
        if prompt:
            text_emb_cond = self.get_text_embedding(prompt)
        else:
            text_emb_cond = self.null_text_emb
        text_emb_uncond = self.null_text_emb

        # Initialize from noise
        latent = torch.randn_like(ref1_latent)

        # Set up scheduler
        self.scheduler.set_timesteps(num_steps)

        for t in tqdm(self.scheduler.timesteps, desc="Sampling"):
            t_batch = t.unsqueeze(0).to(self.device)

            # Unconditional prediction
            noise_uncond = self.forward(
                latent, t_batch,
                ref1_latent, ref2_latent,
                plucker_ref1, plucker_ref2, plucker_target,
                text_emb_uncond
            )

            # Conditional prediction
            noise_cond = self.forward(
                latent, t_batch,
                ref1_latent, ref2_latent,
                plucker_ref1, plucker_ref2, plucker_target,
                text_emb_cond
            )

            # CFG
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            # Scheduler step
            latent = self.scheduler.step(noise_pred, t, latent).prev_sample

        # Decode
        return self.decode_latent(latent)


def train(
    model: PoseConditionedSD,
    dataset: MegaScenesDataset,
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-5,
    save_every: int = 10,
    prompt: str = "",
):
    """Training loop."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            loss = model.training_step(batch, prompt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")

        if (epoch + 1) % save_every == 0:
            torch.save(model.unet.state_dict(), output_path / f"unet_epoch_{epoch+1}.pt")
            print(f"Saved checkpoint to {output_path / f'unet_epoch_{epoch+1}.pt'}")

    # Save final
    torch.save(model.unet.state_dict(), output_path / "unet_final.pt")
    print(f"Training complete. Saved to {output_path / 'unet_final.pt'}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", required=True, help="Scene directories")
    p.add_argument("--output", required=True, help="Output directory for checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--prompt", default="", help="Optional text prompt for training")
    args = p.parse_args()

    print("Loading model...")
    model = PoseConditionedSD()

    print("Loading dataset...")
    dataset = MegaScenesDataset(args.scenes)
    print(f"Found {len(dataset)} training triplets")

    print("Starting training...")
    train(model, dataset, args.output, args.epochs, args.batch_size, args.lr, prompt=args.prompt)
