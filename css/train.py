"""Training script for Pose-Conditioned SD.

Fixes applied:
- LR scheduler: linear warmup + cosine annealing
- EMA of trainable parameters, used for sampling
- Checkpoint saves/restores optimizer, LR scheduler, EMA state
- Resume correctly restores all training state including optimizer momentum
- Proper epoch/step accounting on resume
"""

import argparse
import math
import re
import signal
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from css.data.dataset import MegaScenesDataset, load_image_name_set
from css.models.pose_conditioned_sd import (
    EMAModel,
    PoseConditionedSD,
    load_pose_sd_checkpoint,
    save_pose_sd_checkpoint,
)

_save_requested = False


def _handle_save_signal(signum, frame):
    global _save_requested
    _save_requested = True
    print("\n[Signal] Save requested, will save after current step...")


signal.signal(signal.SIGUSR1, _handle_save_signal)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def _get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    best_epoch, best_step, best_path = -1, 0, None
    for ckpt in output_dir.glob("unet_*.pt"):
        m = re.search(r"epoch[_]?(\d+)", ckpt.stem)
        if m:
            epoch = int(m.group(1))
            step_m = re.search(r"step(\d+)", ckpt.stem)
            step = int(step_m.group(1)) if step_m else 0
            if epoch > best_epoch or (epoch == best_epoch and step > best_step):
                best_epoch, best_step, best_path = epoch, step, ckpt
    return best_path


def _cleanup_old_checkpoints(output_dir: Path, keep_latest: int = 5) -> None:
    """Keep only the N most recent periodic epoch checkpoints."""
    checkpoints = []
    for ckpt in output_dir.glob("unet_epoch_*.pt"):
        m = re.search(r"epoch_(\d+)", ckpt.stem)
        if m:
            checkpoints.append((int(m.group(1)), ckpt))
    checkpoints.sort(key=lambda x: x[0])
    for _, path in checkpoints[:-keep_latest]:
        print(f"Removing old checkpoint: {path.name}")
        path.unlink()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_uint8(t: torch.Tensor) -> np.ndarray:
    return ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()


def _read_lines(path: str | None) -> list[str]:
    if path is None:
        return []
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _resolve_scenes(args) -> list[str]:
    merged = list(args.scenes or []) + _read_lines(args.scenes_file)
    if not merged:
        raise ValueError("Provide at least one scene via --scenes or --scenes-file")
    scenes = list(OrderedDict.fromkeys(merged))
    max_scenes = getattr(args, "max_scenes", None)
    if max_scenes is not None and len(scenes) > max_scenes:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(scenes), max_scenes, replace=False)
        indices.sort()
        scenes = [scenes[i] for i in indices]
    return scenes


# ---------------------------------------------------------------------------
# Sampling with EMA
# ---------------------------------------------------------------------------

def _log_sample(model, ema, trainable_params, dataset, prompt, step, cfg_scale):
    try:
        sample = dataset[0]
        model.eval()

        # Use EMA weights for sampling
        if ema is not None:
            ema.apply_shadow(trainable_params)

        sample_prompt = sample.get("prompt", prompt)
        generated = model.sample(
            sample["ref1_img"].unsqueeze(0),
            sample["ref2_img"].unsqueeze(0),
            sample["plucker_ref1"].unsqueeze(0),
            sample["plucker_ref2"].unsqueeze(0),
            prompt=sample_prompt, num_steps=50, cfg_scale=cfg_scale,
        )
        grid = np.concatenate([
            _to_uint8(sample["ref1_img"]),
            _to_uint8(sample["ref2_img"]),
            _to_uint8(sample["target_img"]),
            _to_uint8(generated[0]),
        ], axis=1)
        wandb.log({"samples/comparison": wandb.Image(grid, caption="ref1 | ref2 | GT | generated")}, step=step)

    except Exception as e:
        print(f"[Warning] Sample generation failed: {e}")
    finally:
        # Restore live weights
        if ema is not None:
            ema.restore(trainable_params)
        model.train()


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_checkpoint(model, optimizer, lr_scheduler, ema, epoch, global_step, ckpt_path):
    save_pose_sd_checkpoint(
        model, ckpt_path,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        ema=ema,
        epoch=epoch,
        global_step=global_step,
    )
    print(f"Saved {ckpt_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model,
    dataset,
    output_dir,
    num_epochs=100,
    batch_size=4,
    lr=1e-5,
    warmup_steps=500,
    save_every=4,
    keep_checkpoints=5,
    prompt="",
    cond_drop_prob=0.1,
    sample_cfg_scale=7.5,
    min_timestep=0,
    max_timestep=None,
    num_workers=0,
    ema_decay=0.9999,
):
    global _save_requested
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    if len(dataloader) == 0:
        raise ValueError("No training batches available. Check scene/split filters and triplet constraints.")

    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # LR schedule: linear warmup + cosine decay
    total_training_steps = num_epochs * len(dataloader)
    lr_scheduler = _get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # EMA
    ema = EMAModel(trainable_params, decay=ema_decay)

    # --- Resume ---
    start_epoch, global_step = 0, 0
    ckpt_path = _find_latest_checkpoint(output_path)
    if ckpt_path is not None:
        print(f"Resuming from {ckpt_path}")
        meta = load_pose_sd_checkpoint(
            model, ckpt_path, str(model.device),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
        )
        start_epoch = meta["epoch"]
        global_step = meta["global_step"]
        print(f"  Restored epoch={start_epoch}, global_step={global_step}")
        if start_epoch >= num_epochs:
            print(f"Latest checkpoint is epoch {start_epoch}; requested --epochs={num_epochs}. Nothing to train.")
            return
    else:
        print("Starting from scratch")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss = 0.0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                optimizer.zero_grad()
                batch_prompt = batch["prompt"] if "prompt" in batch else prompt
                loss = model.training_step(
                    batch,
                    batch_prompt,
                    cond_drop_prob=cond_drop_prob,
                    min_timestep=min_timestep,
                    max_timestep=max_timestep,
                )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                ema.update(trainable_params)

                total_loss += loss.item()
                global_step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

                wandb.log({
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                }, step=global_step)

                if _save_requested:
                    _save_checkpoint(
                        model, optimizer, lr_scheduler, ema,
                        epoch + 1, global_step,
                        output_path / f"unet_signal_epoch{epoch+1}_step{global_step}.pt",
                    )
                    _save_requested = False

            avg_loss = total_loss / len(dataloader)
            wandb.log({"train/epoch_loss": avg_loss, "train/epoch": epoch + 1}, step=global_step)
            print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")

            if (epoch + 1) % save_every == 0:
                _save_checkpoint(
                    model, optimizer, lr_scheduler, ema,
                    epoch + 1, global_step,
                    output_path / f"unet_epoch_{epoch+1}.pt",
                )
                _cleanup_old_checkpoints(output_path, keep_latest=keep_checkpoints)

            _log_sample(model, ema, trainable_params, dataset, prompt, global_step, cfg_scale=sample_cfg_scale)

    except KeyboardInterrupt:
        print("\n[Interrupted] Saving emergency checkpoint...")
        _save_checkpoint(
            model, optimizer, lr_scheduler, ema,
            epoch + 1, global_step,
            output_path / f"unet_interrupted_epoch{epoch+1}_step{global_step}.pt",
        )
        return

    _save_checkpoint(
        model, optimizer, lr_scheduler, ema,
        num_epochs, global_step,
        output_path / "unet_final.pt",
    )
    print("Training complete.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--scenes-file", type=str, default=None)
    p.add_argument("--max-scenes", type=int, default=None, help="Use at most N scenes (randomly sampled if more available)")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=500, help="Linear LR warmup steps")
    p.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay rate")
    p.add_argument("--prompt", default="a photo of the Mysore palace")
    p.add_argument("--prompt-template", type=str, default=None, help='Per-sample prompt template, e.g. "a photo of {scene}"')
    p.add_argument("--max-pair-dist", type=float, default=2.0)
    p.add_argument("--min-dir-sim", type=float, default=0.3)
    p.add_argument("--min-ref-spacing", type=float, default=0.3)
    p.add_argument("--max-triplets", type=int, default=10000)
    p.add_argument("--save-every", type=int, default=4)
    p.add_argument("--keep-checkpoints", type=int, default=5)
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--wandb-project", default="css-pose-sd")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-id", default=None)
    p.add_argument("--cond-drop-prob", type=float, default=0.1)
    p.add_argument("--sample-cfg-scale", type=float, default=7.5)
    p.add_argument("--min-timestep", type=int, default=0)
    p.add_argument("--max-timestep", type=int, default=None)
    p.add_argument("--unet-train-mode", choices=["full", "cond"], default="full")
    p.add_argument("--exclude-image-list", type=str, default=None)
    p.add_argument("--target-include-image-list", type=str, default=None)
    p.add_argument("--reference-include-image-list", type=str, default=None)
    args = p.parse_args()

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        id=args.wandb_id,
        resume="allow",
        config=vars(args),
        settings=wandb.Settings(x_stats_sampling_interval=10),
    )
    wandb.run.log_code("css")

    print("Loading model...")
    model = PoseConditionedSD()
    model.configure_trainable(args.unet_train_mode)
    trainable_params = model.get_trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable_params)
    print(
        f"UNet train mode: {args.unet_train_mode} "
        f"(trainable params incl. ref_encoder: {n_trainable:,})"
    )

    print("Loading dataset...")
    exclude_image_names = load_image_name_set(args.exclude_image_list)
    target_include_image_names = load_image_name_set(args.target_include_image_list)
    reference_include_image_names = load_image_name_set(args.reference_include_image_list)

    if exclude_image_names is not None:
        print(f"Excluding {len(exclude_image_names)} images from all roles")
    if target_include_image_names is not None:
        print(f"Restricting targets to {len(target_include_image_names)} images")
    if reference_include_image_names is not None:
        print(f"Restricting references to {len(reference_include_image_names)} images")

    scenes = _resolve_scenes(args)
    print(f"Using {len(scenes)} scenes")
    if args.prompt_template is not None:
        print(f'Using prompt template: "{args.prompt_template}"')

    dataset = MegaScenesDataset(
        scenes, H=args.H, W=args.W,
        max_pair_distance=args.max_pair_dist,
        max_triplets_per_scene=args.max_triplets,
        min_dir_similarity=args.min_dir_sim,
        min_ref_spacing=args.min_ref_spacing,
        exclude_image_names=exclude_image_names,
        target_include_image_names=target_include_image_names,
        reference_include_image_names=reference_include_image_names,
        prompt_template=args.prompt_template,
    )
    print(f"Found {len(dataset)} training triplets")
    if len(dataset) == 0:
        raise ValueError("Dataset has 0 triplets. Relax triplet constraints or adjust split files.")

    print("Starting training...")
    train(
        model,
        dataset,
        args.output,
        args.epochs,
        args.batch_size,
        args.lr,
        warmup_steps=args.warmup_steps,
        save_every=args.save_every,
        keep_checkpoints=args.keep_checkpoints,
        prompt=args.prompt,
        cond_drop_prob=args.cond_drop_prob,
        sample_cfg_scale=args.sample_cfg_scale,
        min_timestep=args.min_timestep,
        max_timestep=args.max_timestep,
        num_workers=args.num_workers,
        ema_decay=args.ema_decay,
    )

    wandb.finish()


if __name__ == "__main__":
    main()