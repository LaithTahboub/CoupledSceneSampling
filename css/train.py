"""Training script for Pose-Conditioned SD."""

import argparse
import re
import signal
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from css.data.dataset import MegaScenesDataset, load_image_name_set
from css.models.pose_conditioned_sd import (
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


def _find_latest_checkpoint(output_dir: Path) -> tuple[Path | None, int, int]:
    best_epoch, best_step, best_path = -1, 0, None
    for ckpt in output_dir.glob("unet_*.pt"):
        m = re.search(r"epoch[_]?(\d+)", ckpt.stem)
        if m:
            epoch = int(m.group(1))
            step_m = re.search(r"step(\d+)", ckpt.stem)
            step = int(step_m.group(1)) if step_m else 0
            if epoch > best_epoch or (epoch == best_epoch and step > best_step):
                best_epoch, best_step, best_path = epoch, step, ckpt
    return best_path, best_epoch, best_step


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    return ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()


def _log_sample(model, dataset, prompt, step, cfg_scale):
    try:
        sample = dataset[0]
        model.eval()
        generated = model.sample(
            sample["ref1_img"].unsqueeze(0),
            sample["ref2_img"].unsqueeze(0),
            sample["plucker_ref1"].unsqueeze(0),
            sample["plucker_ref2"].unsqueeze(0),
            sample["plucker_target"].unsqueeze(0),
            prompt=prompt, num_steps=50, cfg_scale=cfg_scale,
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
        model.train()


def _save_checkpoint(model, ckpt_path):
    save_pose_sd_checkpoint(model, ckpt_path)
    print(f"Saved {ckpt_path}")


def train(
    model,
    dataset,
    output_dir,
    num_epochs=100,
    batch_size=4,
    lr=1e-5,
    save_every=5,
    prompt="",
    cond_drop_prob=0.1,
    sample_cfg_scale=2.0,
    min_timestep=0,
    max_timestep=None,
):
    global _save_requested
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    if len(dataloader) == 0:
        raise ValueError("No training batches available. Check scene/split filters and triplet constraints.")
    trained_params = [p for p in model.unet.parameters() if p.requires_grad] + list(model.ref_encoder.parameters())
    optimizer = torch.optim.AdamW(trained_params, lr=lr)

    start_epoch, global_step = 0, 0
    ckpt_path, resume_epoch, resume_step = _find_latest_checkpoint(output_path)
    if ckpt_path is not None:
        print(f"Resuming from {ckpt_path} (epoch {resume_epoch}, step {resume_step})")
        load_pose_sd_checkpoint(model, ckpt_path, model.device)
        start_epoch = resume_epoch
        global_step = resume_step
    else:
        print("Starting from scratch")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                optimizer.zero_grad()
                loss = model.training_step(
                    batch,
                    prompt,
                    cond_drop_prob=cond_drop_prob,
                    min_timestep=min_timestep,
                    max_timestep=max_timestep,
                )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trained_params, max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                global_step += 1
                pbar.set_postfix(loss=loss.item())

                wandb.log({
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm.item(),
                }, step=global_step)

                if _save_requested:
                    _save_checkpoint(model, output_path / f"unet_signal_epoch{epoch+1}_step{global_step}.pt")
                    _save_requested = False

            avg_loss = total_loss / len(dataloader)
            wandb.log({"train/epoch_loss": avg_loss, "train/epoch": epoch + 1}, step=global_step)
            print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")

            if (epoch + 1) % save_every == 0:
                _save_checkpoint(model, output_path / f"unet_epoch_{epoch+1}.pt")

            _log_sample(model, dataset, prompt, global_step, cfg_scale=sample_cfg_scale)

    except KeyboardInterrupt:
        print("\n[Interrupted] Saving emergency checkpoint...")
        _save_checkpoint(model, output_path / f"unet_interrupted_epoch{epoch+1}_step{global_step}.pt")
        return

    _save_checkpoint(model, output_path / "unet_final.pt")
    print("Training complete.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--prompt", default="a photo of the Mysore palace")
    p.add_argument("--max-pair-dist", type=float, default=2.0)
    p.add_argument("--max-triplets", type=int, default=10000)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)
    p.add_argument("--wandb-project", default="css-pose-sd")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-id", default=None)
    p.add_argument("--cond-drop-prob", type=float, default=0.1)
    p.add_argument("--sample-cfg-scale", type=float, default=1.0)
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
    n_trainable = sum(p.numel() for p in model.unet.parameters() if p.requires_grad) + \
        sum(p.numel() for p in model.ref_encoder.parameters())
    print(f"UNet train mode: {args.unet_train_mode} (trainable params incl. ref_encoder: {n_trainable:,})")

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

    dataset = MegaScenesDataset(
        args.scenes, H=args.H, W=args.W,
        max_pair_distance=args.max_pair_dist,
        max_triplets_per_scene=args.max_triplets,
        exclude_image_names=exclude_image_names,
        target_include_image_names=target_include_image_names,
        reference_include_image_names=reference_include_image_names,
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
        args.save_every,
        args.prompt,
        cond_drop_prob=args.cond_drop_prob,
        sample_cfg_scale=args.sample_cfg_scale,
        min_timestep=args.min_timestep,
        max_timestep=args.max_timestep,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
