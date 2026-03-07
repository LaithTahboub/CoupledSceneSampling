"""Single-reference debug experiment: one ref image, no pluckers, cross-view attention."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

from css.data.colmap_reader import read_scene
from css.data.dataset import (
    clean_scene_prompt_name,
    load_image_tensor,
    read_scene_prompt_name,
)
from css.data.iou import compute_covisibility
from css.models.cross_view_attention import CrossViewAttention


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _read_lines(path: str | None) -> list[str]:
    if path is None:
        return []
    lines: list[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(s)
    return lines


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    return ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()


@dataclass
class PairRecord:
    scene_name: str
    images_dir: Path
    ref_name: str
    target_name: str
    prompt: str
    covisibility: float
    distance: float


class SingleRefPairDataset(Dataset):
    """Builds (reference, target) image pairs filtered by co-visibility."""

    def __init__(
        self,
        scene_dirs: list[str],
        *,
        H: int = 512,
        W: int = 512,
        min_covisibility: float = 0.10,
        max_covisibility: float = 0.80,
        max_pairs_per_scene: int = 128,
    ):
        self.H = H
        self.W = W
        self.pairs: list[PairRecord] = []

        for scene_spec in scene_dirs:
            scene_dir = Path(scene_spec)
            pairs = self._build_scene_pairs(
                scene_dir, min_covisibility, max_covisibility, max_pairs_per_scene,
            )
            self.pairs.extend(pairs)

        if self.pairs:
            print(f"SingleRefPairDataset: {len(self.pairs)} pairs from {len(scene_dirs)} scenes")
        else:
            print("SingleRefPairDataset: 0 valid pairs found.")

    def _build_scene_pairs(
        self, scene_dir: Path,
        min_covis: float, max_covis: float,
        max_pairs: int,
    ) -> list[PairRecord]:
        cameras, images = read_scene(scene_dir)
        images_dir = scene_dir / "images"
        scene_name = scene_dir.name
        prompt = f"a photo of {clean_scene_prompt_name(read_scene_prompt_name(scene_dir))}"

        # Resolve image names to actual files on disk
        disk_files: set[str] = set()
        for p in images_dir.rglob("*"):
            if p.is_file():
                disk_files.add(p.relative_to(images_dir).as_posix())

        valid = []
        resolved: dict[int, str] = {}
        for img in images.values():
            name = img.name.replace("\\", "/")
            if name in disk_files:
                resolved[img.id] = name
                valid.append(img)

        if len(valid) < 2:
            return []

        positions = {img.id: img.c2w[:3, 3].astype(np.float64) for img in valid}

        candidates: list[PairRecord] = []
        for i, ref in enumerate(valid):
            for tgt in valid[i + 1:]:
                covis = compute_covisibility(ref, tgt)
                if covis < min_covis or covis > max_covis:
                    continue

                dist = float(np.linalg.norm(positions[ref.id] - positions[tgt.id]))
                candidates.append(PairRecord(
                    scene_name=scene_name, images_dir=images_dir,
                    ref_name=resolved[ref.id], target_name=resolved[tgt.id],
                    prompt=prompt, covisibility=covis, distance=dist,
                ))

        if len(candidates) <= max_pairs:
            if candidates:
                cv = [c.covisibility for c in candidates]
                print(f"  {scene_name}: {len(candidates)} pairs, covis=[{min(cv):.3f}, {max(cv):.3f}]")
            return candidates

        # Diverse selection: round-robin through targets so different target
        # images are represented as evenly as possible.
        by_target: dict[str, list[PairRecord]] = {}
        for pair in candidates:
            by_target.setdefault(pair.target_name, []).append(pair)

        # Within each target, sort by covisibility (best first)
        for pairs in by_target.values():
            pairs.sort(key=lambda p: p.covisibility, reverse=True)

        # Shuffle target order for fairness across scenes
        rng = np.random.default_rng(42)
        targets_order = list(by_target.keys())
        rng.shuffle(targets_order)

        selected: list[PairRecord] = []
        round_idx = 0
        while len(selected) < max_pairs:
            added = False
            for tgt in targets_order:
                group = by_target[tgt]
                if round_idx < len(group):
                    selected.append(group[round_idx])
                    added = True
                    if len(selected) >= max_pairs:
                        break
            if not added:
                break
            round_idx += 1

        if selected:
            cv = [c.covisibility for c in selected]
            n_targets = len(set(p.target_name for p in selected))
            print(f"  {scene_name}: {len(selected)} pairs ({n_targets} unique targets), covis=[{min(cv):.3f}, {max(cv):.3f}]")
        return selected

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        rec = self.pairs[idx]
        ref_img, _, _ = load_image_tensor(rec.images_dir, rec.ref_name, self.H, self.W)
        target_img, _, _ = load_image_tensor(rec.images_dir, rec.target_name, self.H, self.W)
        return {
            "ref_img": ref_img,
            "target_img": target_img,
            "prompt": rec.prompt,
            "scene_name": rec.scene_name,
            "ref_name": rec.ref_name,
            "target_name": rec.target_name,
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


class SingleRefSD(nn.Module):
    """2-view CAT3D-style debug model: [target, ref], no Plucker channels."""

    NUM_VIEWS = 2
    PER_VIEW_CH = 5  # latent(4) + mask(1)

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

    def _pack_views(self, ref_lat: torch.Tensor, tgt_lat: torch.Tensor,
                    ref_keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, _, h, w = tgt_lat.shape
        m_ref = torch.ones((b, 1, h, w), device=tgt_lat.device, dtype=tgt_lat.dtype)
        m_tgt = torch.zeros((b, 1, h, w), device=tgt_lat.device, dtype=tgt_lat.dtype)

        if ref_keep_mask is not None:
            keep = ref_keep_mask.to(device=tgt_lat.device, dtype=tgt_lat.dtype).view(b, 1, 1, 1)
            ref_lat = ref_lat * keep
            m_ref = m_ref * keep

        v_tgt = torch.cat([tgt_lat, m_tgt], dim=1)
        v_ref = torch.cat([ref_lat, m_ref], dim=1)
        # Stack along dim=1 so views are interleaved: [tgt0, ref0, tgt1, ref1, ...]
        # This matches CrossViewAttention's rearrange("(bs v) s c -> bs (v s) c")
        return torch.stack([v_tgt, v_ref], dim=1).reshape(b * self.NUM_VIEWS, self.PER_VIEW_CH, h, w)

    def _predict_target_eps(self, packed: torch.Tensor, timesteps: torch.Tensor,
                            text_emb: torch.Tensor, batch_size: int) -> torch.Tensor:
        t = timesteps.repeat_interleave(self.NUM_VIEWS)
        te = text_emb.repeat_interleave(self.NUM_VIEWS, dim=0)
        pred = self.unet(packed, t, encoder_hidden_states=te).sample
        return pred.view(batch_size, self.NUM_VIEWS, *pred.shape[1:])[:, 0]

    def training_step(self, batch: dict, *, cond_drop_prob: float = 0.2) -> torch.Tensor:
        ref_img = batch["ref_img"].to(self.device)
        target_img = batch["target_img"].to(self.device)
        prompts = list(batch["prompt"])
        b = target_img.shape[0]

        ref_lat = self.encode_image(ref_img)
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

        packed = self._pack_views(ref_lat, tgt_lat_noisy, ref_keep_mask)
        pred = self._predict_target_eps(packed, timesteps, text_emb, batch_size=b)

        target = noise  # epsilon prediction
        return F.mse_loss(pred.float(), target.float())

    @torch.inference_mode()
    def sample(self, *, ref_img: torch.Tensor, prompt: str,
               num_steps: int = 40, cfg_scale: float = 4.0, seed: int = 42) -> torch.Tensor:
        ref_lat = self.encode_image(ref_img.to(self.device))
        b = ref_lat.shape[0]
        text_cond = self.get_text_embeddings([prompt]).expand(b, -1, -1)
        text_uncond = self.null_text_emb.expand(b, -1, -1)

        self.scheduler.set_timesteps(num_steps)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        latent = torch.randn(ref_lat.shape, generator=gen, device=self.device, dtype=ref_lat.dtype)

        for t in self.scheduler.timesteps:
            t_b = torch.full((b,), int(t), device=self.device, dtype=torch.long)
            latent_in = self.scheduler.scale_model_input(latent, t)

            packed_cond = self._pack_views(ref_lat, latent_in)
            eps_cond = self._predict_target_eps(packed_cond, t_b, text_cond, b)

            keep_none = torch.zeros(b, device=self.device, dtype=torch.bool)
            packed_uncond = self._pack_views(ref_lat, latent_in, keep_none)
            eps_uncond = self._predict_target_eps(packed_uncond, t_b, text_uncond, b)

            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            latent = self.scheduler.step(eps, t, latent).prev_sample

        return self.decode_latent(latent)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def _save_checkpoint(model: SingleRefSD, optimizer: torch.optim.Optimizer,
                     epoch: int, global_step: int, path: Path) -> None:
    torch.save({
        "unet": model.unet.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, path)
    print(f"Saved checkpoint: {path}")


def _cleanup_checkpoints(output_dir: Path, keep: int = 3) -> None:
    ckpts = sorted(output_dir.glob("unet_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    for p in ckpts[:-keep]:
        print(f"Removing old checkpoint: {p.name}")
        p.unlink()


# ---------------------------------------------------------------------------
# Sampling & logging
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _log_sample(model: SingleRefSD, dataset: Dataset, idx: int, tag: str,
                num_steps: int, cfg_scale: float, seed: int, step: int) -> None:
    """Generate one sample and log [ref | target | generated] strip to wandb."""
    item = dataset[idx]
    ref = item["ref_img"].unsqueeze(0)
    generated = model.sample(ref_img=ref, prompt=item["prompt"],
                             num_steps=num_steps, cfg_scale=cfg_scale, seed=seed)[0].cpu()

    strip = np.concatenate([_to_uint8(item["ref_img"]), _to_uint8(item["target_img"]),
                            _to_uint8(generated)], axis=1)
    caption = f'{item["scene_name"]} | ref={item["ref_name"]} | tgt={item["target_name"]}'
    wandb.log({f"samples/{tag}": wandb.Image(strip, caption=caption)}, step=step)


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def _build_split(
    pairs: list[PairRecord],
    seed: int,
    test_scenes_pct: float,
    test_targets_per_scene: int,
) -> tuple[list[int], list[int], dict]:
    """Split pairs into train/test indices.

    Test set = all pairs from fully withheld scenes
             + pairs whose target is a withheld target in train scenes.

    Returns (train_indices, test_indices, split_info_dict).
    """
    rng = np.random.default_rng(seed)

    # All unique scene names
    scene_names = sorted(set(p.scene_name for p in pairs))
    n_test_scenes = max(0, int(round(len(scene_names) * test_scenes_pct / 100)))

    # Pick test scenes
    perm = rng.permutation(len(scene_names))
    test_scene_set = set(scene_names[i] for i in perm[:n_test_scenes])
    train_scene_set = set(scene_names) - test_scene_set

    # For each train scene, withhold some target images
    test_targets_by_scene: dict[str, list[str]] = {}
    for sn in sorted(train_scene_set):
        scene_pairs = [p for p in pairs if p.scene_name == sn]
        target_names = sorted(set(p.target_name for p in scene_pairs))
        n_hold = min(test_targets_per_scene, max(0, len(target_names) - 1))
        if n_hold > 0:
            rng_scene = np.random.default_rng(seed + hash(sn) % (2**31))
            held = rng_scene.choice(target_names, size=n_hold, replace=False).tolist()
            test_targets_by_scene[sn] = held

    # Partition pairs
    withheld_targets_lookup = {
        sn: set(tgts) for sn, tgts in test_targets_by_scene.items()
    }
    train_indices: list[int] = []
    test_indices: list[int] = []
    for i, p in enumerate(pairs):
        if p.scene_name in test_scene_set:
            test_indices.append(i)
        elif p.target_name in withheld_targets_lookup.get(p.scene_name, set()):
            test_indices.append(i)
        else:
            train_indices.append(i)

    split_info = {
        "seed": seed,
        "test_scenes_pct": test_scenes_pct,
        "test_targets_per_scene": test_targets_per_scene,
        "test_scenes": sorted(test_scene_set),
        "train_scenes": sorted(train_scene_set),
        "withheld_targets_by_scene": {
            sn: sorted(tgts) for sn, tgts in test_targets_by_scene.items()
        },
        "num_train_pairs": len(train_indices),
        "num_test_pairs": len(test_indices),
    }
    return train_indices, test_indices, split_info


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--scenes-file", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--split-dir", type=str, default=None)

    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--train-mode", choices=["cond", "full"], default="cond")
    p.add_argument("--pretrained-model", type=str, default="manojb/stable-diffusion-2-1-base")
    p.add_argument("--cond-drop-prob", type=float, default=0.20)
    p.add_argument("--gradient-checkpointing", action="store_true")

    p.add_argument("--min-covisibility", type=float, default=0.10)
    p.add_argument("--max-covisibility", type=float, default=0.80)
    p.add_argument("--max-pairs-per-scene", type=int, default=128)

    # Split config
    p.add_argument("--test-scenes-pct", type=float, default=5.0,
                    help="Percent of scenes to fully withhold for testing")
    p.add_argument("--test-targets-per-scene", type=int, default=1,
                    help="Number of target images to withhold per train scene")

    # Checkpoint config
    p.add_argument("--save-every", type=int, default=7)
    p.add_argument("--keep-checkpoints", type=int, default=3)

    p.add_argument("--sample-steps", type=int, default=40)
    p.add_argument("--sample-cfg-scale", type=float, default=4.0)

    p.add_argument("--wandb-project", type=str, default="CoupledSceneSampling")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode,
        config=vars(args),
    )

    scenes = list(dict.fromkeys((args.scenes or []) + _read_lines(args.scenes_file)))
    if not scenes:
        raise ValueError("Provide --scenes or --scenes-file")

    dataset = SingleRefPairDataset(
        scene_dirs=scenes, H=args.H, W=args.W,
        min_covisibility=args.min_covisibility, max_covisibility=args.max_covisibility,
        max_pairs_per_scene=args.max_pairs_per_scene,
    )
    if len(dataset) < 2:
        raise ValueError(f"Need >= 2 pairs, got {len(dataset)}")

    # Train/test split: withhold full scenes + specific targets per scene
    train_indices, test_indices, split_info = _build_split(
        dataset.pairs, args.seed, args.test_scenes_pct, args.test_targets_per_scene,
    )
    print(f"Train: {len(train_indices)} pairs | Test: {len(test_indices)} pairs")
    if split_info["test_scenes"]:
        print(f"  Withheld scenes ({len(split_info['test_scenes'])}): {split_info['test_scenes']}")

    # Save split info
    split_dir = Path(args.split_dir) if args.split_dir else output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "split_info.json"
    split_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    print(f"Split saved to {split_path}")

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    if len(train_dataset) == 0:
        raise ValueError("No training pairs after split. Relax constraints.")

    model = SingleRefSD(pretrained_model=args.pretrained_model)
    model.configure_trainable(args.train_mode)
    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
        model.cross_view_attention.attach(model.unet)

    trainable_params = model.get_trainable_parameters()
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    global_step = 0
    try:
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for batch in pbar:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model.training_step(batch, cond_drop_prob=args.cond_drop_prob)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                global_step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                wandb.log({"train/loss": loss.item()}, step=global_step)

            avg_loss = epoch_loss / max(1, len(train_loader))
            print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.5f}")

            # Sample one train + one test image each epoch
            model.eval()
            train_idx = epoch % len(train_dataset)
            test_idx = epoch % max(1, len(test_dataset))
            sample_seed = args.seed + epoch

            _log_sample(model, train_dataset, train_idx, "train",
                        args.sample_steps, args.sample_cfg_scale, sample_seed, global_step)
            if len(test_dataset) > 0:
                _log_sample(model, test_dataset, test_idx, "test",
                            args.sample_steps, args.sample_cfg_scale, sample_seed, global_step)

            # Always save latest
            _save_checkpoint(model, optimizer, epoch + 1, global_step,
                             output_dir / "unet_latest.pt")

            # Save numbered checkpoint every N epochs
            if (epoch + 1) % args.save_every == 0:
                _save_checkpoint(model, optimizer, epoch + 1, global_step,
                                 output_dir / f"unet_epoch_{epoch + 1}.pt")
                _cleanup_checkpoints(output_dir, keep=args.keep_checkpoints)

        # Save final checkpoint
        _save_checkpoint(model, optimizer, args.epochs, global_step,
                         output_dir / "unet_final.pt")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
