"""Training script for Pose-Conditioned SD (CAT3D-style channel concat baseline)."""

from __future__ import annotations

import argparse
import math
import random
import re
import signal
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from css.data.dataset import MegaScenesDataset, load_image_name_set
from css.models.EMA import EMAModel, load_pose_sd_checkpoint, save_pose_sd_checkpoint
from css.models.pose_conditioned_sd import PoseConditionedSD
from css.scene_sampling import build_comparison_grid

_save_requested = False


def _handle_save_signal(signum, frame) -> None:  # pragma: no cover - signal path
    del signum, frame
    global _save_requested
    _save_requested = True
    print("\n[signal] Save requested; writing checkpoint after current optimizer step.")


signal.signal(signal.SIGUSR1, _handle_save_signal)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# LR schedule
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
    best_epoch, best_step, best_path = -1, -1, None
    for ckpt in output_dir.glob("unet_*.pt"):
        m = re.search(r"epoch[_]?(\d+)", ckpt.stem)
        if not m:
            continue
        epoch = int(m.group(1))
        step_m = re.search(r"step(\d+)", ckpt.stem)
        step = int(step_m.group(1)) if step_m else 0
        if epoch > best_epoch or (epoch == best_epoch and step > best_step):
            best_epoch, best_step, best_path = epoch, step, ckpt

    latest_ckpt = output_dir / "unet_latest.pt"
    final_ckpt = output_dir / "unet_final.pt"
    candidates = [p for p in (latest_ckpt, final_ckpt, best_path) if p is not None and p.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _cleanup_old_checkpoints(output_dir: Path, keep_latest: int = 5) -> None:
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
        if args.triplets_manifest:
            return []
        raise ValueError("Provide --scenes/--scenes-file or --triplets-manifest")

    scenes = list(OrderedDict.fromkeys(merged))
    if args.max_scenes is not None and len(scenes) > args.max_scenes:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(scenes), size=args.max_scenes, replace=False))
        scenes = [scenes[i] for i in keep.tolist()]
    return scenes


def _autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    if mixed_precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    if mixed_precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unknown mixed_precision mode: {mixed_precision}")


# ---------------------------------------------------------------------------
# Sampling logger
# ---------------------------------------------------------------------------


def _log_sample(
    model: PoseConditionedSD,
    ema: EMAModel,
    trainable_params: list[torch.nn.Parameter],
    dataset: MegaScenesDataset,
    prompt: str,
    step: int,
    cfg_scale: float,
    num_steps: int,
) -> None:
    try:
        sample_idx = np.random.randint(0, len(dataset))
        sample = dataset[sample_idx]
        model.eval()

        ema.apply_shadow(trainable_params)

        sample_prompt = sample.get("prompt", prompt)
        generated = model.sample(
            sample["ref1_img"].unsqueeze(0),
            sample["ref2_img"].unsqueeze(0),
            sample["plucker_ref1"].unsqueeze(0),
            sample["plucker_ref2"].unsqueeze(0),
            sample["plucker_tgt"].unsqueeze(0),
            prompt=sample_prompt,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
        )
        grid = build_comparison_grid(
            sample["ref1_img"],
            sample["ref2_img"],
            sample["target_img"],
            generated[0],
            prompt=sample_prompt,
        )
        wandb.log(
            {
                "samples/comparison": wandb.Image(
                    grid,
                    caption=f'idx={sample_idx} prompt="{sample_prompt}"',
                )
            },
            step=step,
        )

    except Exception as exc:
        print(f"[warning] sample generation failed: {exc}")
    finally:
        ema.restore(trainable_params)
        model.train()


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def _save_checkpoint(
    model: PoseConditionedSD,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    ema: EMAModel,
    epoch: int,
    global_step: int,
    ckpt_path: Path,
) -> None:
    save_pose_sd_checkpoint(
        model=model,
        ckpt_path=ckpt_path,
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
    model: PoseConditionedSD,
    dataset: MegaScenesDataset,
    output_dir: str,
    *,
    num_epochs: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    grad_accum_steps: int,
    lr: float,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    warmup_steps: int,
    save_every: int,
    keep_checkpoints: int,
    prompt: str,
    cond_drop_prob: float,
    sample_cfg_scale: float,
    sample_steps: int,
    sample_every: int,
    min_timestep: int,
    max_timestep: int | None,
    noise_offset: float,
    min_snr_gamma: float | None,
    ema_decay: float,
    grad_clip: float,
    mixed_precision: str,
    resume_from: str | None,
) -> None:
    global _save_requested

    if save_every <= 0:
        raise ValueError("--save-every must be >= 1")
    if keep_checkpoints <= 0:
        raise ValueError("--keep-checkpoints must be >= 1")
    if sample_every <= 0:
        raise ValueError("--sample-every must be >= 1")
    if grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be >= 1")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": (model.device.type == "cuda"),
        "persistent_workers": (num_workers > 0),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(dataset, **loader_kwargs)
    if len(dataloader) == 0:
        raise ValueError("No training batches available. Check triplet/split constraints.")

    trainable_params = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
        weight_decay=weight_decay,
    )

    steps_per_epoch = math.ceil(len(dataloader) / grad_accum_steps)
    total_training_steps = num_epochs * steps_per_epoch
    lr_scheduler = _get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    ema = EMAModel(trainable_params, decay=ema_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(model.device.type == "cuda" and mixed_precision == "fp16"))

    start_epoch, global_step = 0, 0
    ckpt_path = Path(resume_from) if resume_from else _find_latest_checkpoint(output_path)
    if ckpt_path is not None and ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        meta = load_pose_sd_checkpoint(
            model,
            ckpt_path,
            str(model.device),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
        )
        start_epoch = int(meta["epoch"])
        global_step = int(meta["global_step"])
        print(f"  Restored epoch={start_epoch}, global_step={global_step}")
        if start_epoch >= num_epochs:
            print(f"Checkpoint already at epoch {start_epoch}; requested epochs={num_epochs}. Nothing to do.")
            return
    else:
        print("Starting from scratch")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            micro_count = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for micro_step, batch in enumerate(pbar, start=1):
                batch_prompt = batch["prompt"] if "prompt" in batch else prompt

                with _autocast_context(model.device, mixed_precision):
                    loss = model.training_step(
                        batch,
                        batch_prompt,
                        cond_drop_prob=cond_drop_prob,
                        min_timestep=min_timestep,
                        max_timestep=max_timestep,
                        noise_offset=noise_offset,
                        min_snr_gamma=min_snr_gamma,
                    )

                epoch_loss += loss.item()
                micro_count += 1

                loss_for_backward = loss / grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                should_step = (micro_step % grad_accum_steps == 0) or (micro_step == len(dataloader))
                if not should_step:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                    continue

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                if grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip)
                else:
                    grad_norm_terms = [p.grad.detach().norm() for p in trainable_params if p.grad is not None]
                    if grad_norm_terms:
                        grad_norm = torch.linalg.vector_norm(torch.stack(grad_norm_terms))
                    else:
                        grad_norm = torch.tensor(0.0, device=model.device)
                grad_norm_value = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                ema.update(trainable_params)

                global_step += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/grad_norm": grad_norm_value,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

                if _save_requested:
                    _save_checkpoint(
                        model,
                        optimizer,
                        lr_scheduler,
                        ema,
                        epoch + 1,
                        global_step,
                        output_path / f"unet_signal_epoch{epoch + 1}_step{global_step}.pt",
                    )
                    _save_requested = False

            avg_loss = epoch_loss / max(1, micro_count)
            wandb.log({"train/epoch_loss": avg_loss, "train/epoch": epoch + 1}, step=global_step)
            print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

            # Always keep an up-to-date local checkpoint for restart safety.
            _save_checkpoint(
                model,
                optimizer,
                lr_scheduler,
                ema,
                epoch + 1,
                global_step,
                output_path / "unet_latest.pt",
            )

            if (epoch + 1) % save_every == 0:
                _save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    ema,
                    epoch + 1,
                    global_step,
                    output_path / f"unet_epoch_{epoch + 1}.pt",
                )
                _cleanup_old_checkpoints(output_path, keep_latest=keep_checkpoints)

            if (epoch + 1) % sample_every == 0:
                _log_sample(
                    model,
                    ema,
                    trainable_params,
                    dataset,
                    prompt,
                    global_step,
                    cfg_scale=sample_cfg_scale,
                    num_steps=sample_steps,
                )

    except KeyboardInterrupt:
        print("\n[interrupted] Saving emergency checkpoint...")
        _save_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            ema,
            epoch + 1,
            global_step,
            output_path / "unet_latest.pt",
        )
        _save_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            ema,
            epoch + 1,
            global_step,
            output_path / f"unet_interrupted_epoch{epoch + 1}_step{global_step}.pt",
        )
        return

    if num_epochs % save_every != 0:
        _save_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            ema,
            num_epochs,
            global_step,
            output_path / f"unet_epoch_{num_epochs}.pt",
        )
        _cleanup_old_checkpoints(output_path, keep_latest=keep_checkpoints)

    _save_checkpoint(
        model,
        optimizer,
        lr_scheduler,
        ema,
        num_epochs,
        global_step,
        output_path / "unet_final.pt",
    )
    _save_checkpoint(
        model,
        optimizer,
        lr_scheduler,
        ema,
        num_epochs,
        global_step,
        output_path / "unet_latest.pt",
    )
    print("Training complete.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--scenes-file", type=str, default=None)
    p.add_argument("--max-scenes", type=int, default=None)
    p.add_argument("--triplets-manifest", type=str, default=None, help="Optional packed triplet jsonl")

    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=1)

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--warmup-steps", type=int, default=1000)

    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--prompt", default="a photo of a scene")
    p.add_argument("--prompt-template", type=str, default="a photo of {scene}")

    p.add_argument("--max-pair-dist", type=float, default=2.5)
    p.add_argument("--min-dir-sim", type=float, default=0.2)
    p.add_argument("--min-ref-spacing", type=float, default=0.25)
    p.add_argument("--max-triplets", type=int, default=24)

    p.add_argument("--cond-drop-prob", type=float, default=0.15)
    p.add_argument("--noise-offset", type=float, default=0.05)
    p.add_argument("--min-snr-gamma", type=float, default=5.0)
    p.add_argument("--min-timestep", type=int, default=20)
    p.add_argument("--max-timestep", type=int, default=980)

    p.add_argument("--sample-cfg-scale", type=float, default=7.5)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--sample-every", type=int, default=1)

    p.add_argument("--save-every", type=int, default=4)
    p.add_argument("--keep-checkpoints", type=int, default=5)
    p.add_argument("--resume-from", type=str, default=None)

    p.add_argument("--unet-train-mode", choices=["full", "cond"], default="cond")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--xformers-attention", action="store_true")

    p.add_argument("--exclude-image-list", type=str, default=None)
    p.add_argument("--target-include-image-list", type=str, default=None)
    p.add_argument("--reference-include-image-list", type=str, default=None)

    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)

    p.add_argument("--wandb-project", default="CoupledSceneSampling")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-id", default=None)
    args = p.parse_args()

    _set_seed(args.seed)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        id=args.wandb_id,
        resume="allow",
        config=vars(args),
        settings=wandb.Settings(x_stats_sampling_interval=10),
    )
    if wandb.run is not None:
        wandb.run.log_code("css")

    print("Loading model...")
    model = PoseConditionedSD()
    model.configure_trainable(args.unet_train_mode)
    model.configure_memory_optimizations(
        gradient_checkpointing=args.gradient_checkpointing,
        xformers_attention=args.xformers_attention,
    )
    trainable_params = model.get_trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"UNet train mode: {args.unet_train_mode} (trainable params: {n_trainable:,})")

    print("Loading dataset...")
    exclude_image_names = load_image_name_set(args.exclude_image_list)
    target_include_image_names = load_image_name_set(args.target_include_image_list)
    reference_include_image_names = load_image_name_set(args.reference_include_image_list)

    scenes = _resolve_scenes(args)
    if args.triplets_manifest is not None:
        print(f"Using triplets manifest: {args.triplets_manifest}")
    else:
        print(f"Using {len(scenes)} scenes")

    dataset = MegaScenesDataset(
        scenes,
        H=args.H,
        W=args.W,
        max_pair_distance=args.max_pair_dist,
        max_triplets_per_scene=args.max_triplets,
        min_dir_similarity=args.min_dir_sim,
        min_ref_spacing=args.min_ref_spacing,
        exclude_image_names=exclude_image_names,
        target_include_image_names=target_include_image_names,
        reference_include_image_names=reference_include_image_names,
        prompt_template=args.prompt_template,
        triplets_manifest=args.triplets_manifest,
    )
    print(f"Found {len(dataset)} training triplets")
    if len(dataset) == 0:
        raise ValueError("Dataset has 0 triplets. Relax constraints or adjust split files.")

    print("Starting training...")
    train(
        model,
        dataset,
        args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        warmup_steps=args.warmup_steps,
        save_every=args.save_every,
        keep_checkpoints=args.keep_checkpoints,
        prompt=args.prompt,
        cond_drop_prob=args.cond_drop_prob,
        sample_cfg_scale=args.sample_cfg_scale,
        sample_steps=args.sample_steps,
        sample_every=args.sample_every,
        min_timestep=args.min_timestep,
        max_timestep=args.max_timestep,
        noise_offset=args.noise_offset,
        min_snr_gamma=args.min_snr_gamma,
        ema_decay=args.ema_decay,
        grad_clip=args.grad_clip,
        mixed_precision=args.mixed_precision,
        resume_from=args.resume_from,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
