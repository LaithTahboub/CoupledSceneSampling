"""Training script for PoseSD (3-view, Plucker-conditioned).

Extends the debug_single_ref_experiment pattern to the full model with
two reference images and Plucker ray conditioning.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from css.data.MegaScenesTriplets import MegaScenesTriplets, TripletRecord
from css.models.pose_sd import PoseSD


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


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def _build_split(
    triplets: list[TripletRecord],
    seed: int,
    test_scenes_pct: float,
    test_targets_per_scene: int,
) -> tuple[list[int], list[int], dict]:
    rng = np.random.default_rng(seed)

    scene_names = sorted(set(t.scene_name for t in triplets))
    n_test_scenes = max(0, int(round(len(scene_names) * test_scenes_pct / 100)))

    perm = rng.permutation(len(scene_names))
    test_scene_set = set(scene_names[i] for i in perm[:n_test_scenes])
    train_scene_set = set(scene_names) - test_scene_set

    test_targets_by_scene: dict[str, list[str]] = {}
    for sn in sorted(train_scene_set):
        scene_triplets = [t for t in triplets if t.scene_name == sn]
        target_names = sorted(set(t.target_name for t in scene_triplets))
        n_hold = min(test_targets_per_scene, max(0, len(target_names) - 1))
        if n_hold > 0:
            rng_scene = np.random.default_rng(seed + hash(sn) % (2**31))
            held = rng_scene.choice(target_names, size=n_hold, replace=False).tolist()
            test_targets_by_scene[sn] = held

    withheld_lookup = {sn: set(tgts) for sn, tgts in test_targets_by_scene.items()}
    train_indices: list[int] = []
    test_indices: list[int] = []
    for i, t in enumerate(triplets):
        if t.scene_name in test_scene_set:
            test_indices.append(i)
        elif t.target_name in withheld_lookup.get(t.scene_name, set()):
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
        "num_train_triplets": len(train_indices),
        "num_test_triplets": len(test_indices),
    }
    return train_indices, test_indices, split_info


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def _save_checkpoint(model: PoseSD, optimizer: torch.optim.Optimizer,
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
def _log_sample(model: PoseSD, dataset, idx: int, tag: str,
                num_steps: int, cfg_scale: float, seed: int, step: int) -> None:
    item = dataset[idx]
    generated = model.sample(
        ref1_img=item["ref1_img"].unsqueeze(0),
        ref2_img=item["ref2_img"].unsqueeze(0),
        pl_ref1=item["plucker_ref1"].unsqueeze(0),
        pl_ref2=item["plucker_ref2"].unsqueeze(0),
        pl_tgt=item["plucker_tgt"].unsqueeze(0),
        prompt=item["prompt"],
        num_steps=num_steps, cfg_scale=cfg_scale, seed=seed,
    )[0].cpu()

    # [ref1 | ref2 | target | generated]
    strip = np.concatenate([
        _to_uint8(item["ref1_img"]),
        _to_uint8(item["ref2_img"]),
        _to_uint8(item["target_img"]),
        _to_uint8(generated),
    ], axis=1)
    caption = f'{item["scene_name"]} | r1={item["ref1_name"]} | r2={item["ref2_name"]} | tgt={item["target_name"]}'
    wandb.log({f"samples/{tag}": wandb.Image(strip, caption=caption)}, step=step)


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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--train-mode", choices=["cond", "full"], default="cond")
    p.add_argument("--pretrained-model", type=str, default="manojb/stable-diffusion-2-1-base")
    p.add_argument("--cond-drop-prob", type=float, default=0.15)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--xformers-attention", action="store_true")

    p.add_argument("--min-covisibility", type=float, default=0.15)
    p.add_argument("--max-covisibility", type=float, default=0.55)
    p.add_argument("--min-ref-covisibility", type=float, default=0.10)
    p.add_argument("--max-ref-covisibility", type=float, default=0.70)
    p.add_argument("--min-distance", type=float, default=0.10)
    p.add_argument("--max-triplets-per-scene", type=int, default=64)

    p.add_argument("--test-scenes-pct", type=float, default=5.0)
    p.add_argument("--test-targets-per-scene", type=int, default=1)

    p.add_argument("--save-every", type=int, default=7)
    p.add_argument("--keep-checkpoints", type=int, default=3)

    p.add_argument("--sample-steps", type=int, default=50)
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

    dataset = MegaScenesTriplets(
        scene_dirs=scenes, H=args.H, W=args.W,
        min_covisibility=args.min_covisibility,
        max_covisibility=args.max_covisibility,
        min_ref_covisibility=args.min_ref_covisibility,
        max_ref_covisibility=args.max_ref_covisibility,
        min_distance=args.min_distance,
        max_triplets_per_scene=args.max_triplets_per_scene,
    )
    if len(dataset) < 2:
        raise ValueError(f"Need >= 2 triplets, got {len(dataset)}")

    train_indices, test_indices, split_info = _build_split(
        dataset.triplets, args.seed, args.test_scenes_pct, args.test_targets_per_scene,
    )
    print(f"Train: {len(train_indices)} triplets | Test: {len(test_indices)} triplets")
    if split_info["test_scenes"]:
        print(f"  Withheld scenes ({len(split_info['test_scenes'])}): {split_info['test_scenes']}")

    split_dir = Path(args.split_dir) if args.split_dir else output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "split_info.json"
    split_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    print(f"Split saved to {split_path}")

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    if len(train_dataset) == 0:
        raise ValueError("No training triplets after split. Relax constraints.")

    model = PoseSD(pretrained_model=args.pretrained_model)
    model.configure_trainable(args.train_mode)
    model.configure_memory_optimizations(
        gradient_checkpointing=args.gradient_checkpointing,
        xformers_attention=args.xformers_attention,
    )

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

            # Sample one train + one test triplet each epoch
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
