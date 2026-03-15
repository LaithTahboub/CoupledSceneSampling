"""Training script for PoseSD (3-view, Plucker-conditioned).

Supports:
- Multi-GPU training via DDP (torchrun)
- Step-based training with configurable total steps
- EMA weights
- Cosine LR schedule with warmup
- Gradient accumulation
- Difficulty-bucketed dataset with weighted sampling
- Structured conditioning dropout
- Randomized slot order
- Validation with PSNR/SSIM/LPIPS and reference-copy detection
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from tqdm import tqdm

from css.config import DataConfig, TrainConfig
from css.data.MegaScenesDataset import Difficulty, MegaScenesDataset, SceneRecord
from css.models.EMA import EMAModel, load_pose_sd_checkpoint, save_pose_sd_checkpoint
from css.models.pose_sd import PoseSD
from css.train.validation import ValMetrics, run_validation, to_uint8

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _setup_distributed() -> None:
    if "RANK" not in os.environ:
        return
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(_local_rank())


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    scheduler_type: str = "cosine",
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if scheduler_type == "constant_with_warmup":
            return 1.0
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _compute_bucket_weights(
    dataset: MegaScenesDataset,
    indices: list[int],
    bucket_ratios: dict[Difficulty, float],
) -> list[float]:
    """Compute per-sample weights so that expected sampling matches bucket ratios."""
    bucket_counts: dict[Difficulty, int] = {d: 0 for d in Difficulty}
    idx_difficulties: list[Difficulty] = []
    for i in indices:
        diff = dataset.triplets[i].difficulty
        bucket_counts[diff] += 1
        idx_difficulties.append(diff)

    weights = []
    for diff in idx_difficulties:
        count = bucket_counts[diff]
        ratio = bucket_ratios.get(diff, 0.0)
        if count > 0 and ratio > 0:
            weights.append(ratio / count)
        else:
            weights.append(0.0)
    return weights


def _build_weighted_sampler(
    dataset: MegaScenesDataset,
    indices: list[int],
    bucket_ratios: dict[Difficulty, float],
) -> WeightedRandomSampler:
    """Build a weighted sampler for single-GPU training."""
    weights = _compute_bucket_weights(dataset, indices, bucket_ratios)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(indices),
        replacement=True,
    )


class DistributedWeightedSampler(Sampler[int]):
    """Distributed sampler that respects bucket-ratio weights.

    Each rank gets a disjoint subset of indices (like DistributedSampler),
    but within that subset, samples are drawn according to bucket weights
    (like WeightedRandomSampler).
    """

    def __init__(
        self,
        dataset: MegaScenesDataset,
        indices: list[int],
        bucket_ratios: dict[Difficulty, float],
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.dataset = dataset
        self.all_indices = indices
        self.bucket_ratios = bucket_ratios
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        # Pad to make evenly divisible
        self.total_size = int(math.ceil(len(indices) / num_replicas)) * num_replicas
        self.num_samples = self.total_size // num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        # Deterministic shuffle (same across all ranks for consistent partitioning)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(len(self.all_indices), generator=g).tolist()

        # Pad
        while len(perm) < self.total_size:
            perm.append(perm[len(perm) % len(self.all_indices)])

        # Partition: each rank gets every num_replicas-th element
        # rank_perm contains Subset-local indices (0..len(all_indices)-1)
        rank_perm = perm[self.rank::self.num_replicas]

        # Look up difficulty using the raw dataset index for weight computation
        weights = []
        bucket_counts: dict[Difficulty, int] = {d: 0 for d in Difficulty}
        idx_diffs: list[Difficulty] = []
        for local_idx in rank_perm:
            dataset_idx = self.all_indices[local_idx]
            diff = self.dataset.triplets[dataset_idx].difficulty
            bucket_counts[diff] += 1
            idx_diffs.append(diff)

        for diff in idx_diffs:
            count = bucket_counts[diff]
            ratio = self.bucket_ratios.get(diff, 0.0)
            if count > 0 and ratio > 0:
                weights.append(ratio / count)
            else:
                weights.append(0.0)

        # Weighted sampling within this rank's partition
        weights_t = torch.tensor(weights, dtype=torch.double)
        if weights_t.sum() < 1e-12:
            sample_order = torch.randperm(len(rank_perm), generator=g).tolist()
        else:
            sample_order = torch.multinomial(
                weights_t, num_samples=self.num_samples, replacement=True,
                generator=g,
            ).tolist()

        # Yield Subset-local indices (not raw dataset indices)
        yield from (rank_perm[s] for s in sample_order)


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def _build_split(
    triplets: list[SceneRecord],
    seed: int,
    test_scenes_pct: float,
    test_targets_per_scene: int,
) -> tuple[list[int], list[int], list[int], dict]:
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
    withheld_target_indices: list[int] = []
    for i, t in enumerate(triplets):
        if t.scene_name in test_scene_set:
            test_indices.append(i)
        elif t.target_name in withheld_lookup.get(t.scene_name, set()):
            test_indices.append(i)
            withheld_target_indices.append(i)
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
        "num_withheld_target_triplets": len(withheld_target_indices),
    }
    return train_indices, test_indices, withheld_target_indices, split_info


# ---------------------------------------------------------------------------
# Validation & logging
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _log_validation(
    model: PoseSD,
    val_dataset,
    val_indices: list[int],
    global_step: int,
    cfg: TrainConfig,
) -> None:
    """Run validation and log metrics + sample images."""
    if not _is_main_process():
        return

    model.eval()

    # Select a diverse subset of validation samples
    max_val = min(8, len(val_indices))
    # Try to get samples from each bucket
    bucket_indices: dict[str, list[int]] = {}
    for i in val_indices:
        item = val_dataset[i]
        diff = item.get("difficulty", "unknown") if isinstance(item, dict) else "unknown"
        bucket_indices.setdefault(diff, []).append(i)

    selected = []
    for diff, idxs in bucket_indices.items():
        n_from_bucket = max(1, max_val // max(1, len(bucket_indices)))
        selected.extend(idxs[:n_from_bucket])
    selected = selected[:max_val]

    if not selected:
        return

    metrics, per_sample = run_validation(
        model, val_dataset, selected,
        num_steps=cfg.val_sample_steps,
        cfg_scale=cfg.val_cfg_scale,
        seed=cfg.seed,
        max_samples=max_val,
        compute_lpips=True,
    )

    if not _WANDB_AVAILABLE:
        print(f"  Val step {global_step}: PSNR={metrics.psnr_mean:.2f} SSIM={metrics.ssim_mean:.4f} "
              f"LPIPS={metrics.lpips_mean:.4f} CopyRatio={metrics.copy_ratio_mean:.3f}")
        return

    log_dict = {
        "val/psnr": metrics.psnr_mean,
        "val/ssim": metrics.ssim_mean,
        "val/lpips": metrics.lpips_mean,
        "val/copy_ratio": metrics.copy_ratio_mean,
    }

    # Per-bucket metrics
    if metrics.bucket_psnr:
        for diff, val in metrics.bucket_psnr.items():
            log_dict[f"val/psnr_{diff}"] = val
    if metrics.bucket_lpips:
        for diff, val in metrics.bucket_lpips.items():
            log_dict[f"val/lpips_{diff}"] = val
    if metrics.bucket_copy_ratio:
        for diff, val in metrics.bucket_copy_ratio.items():
            log_dict[f"val/copy_ratio_{diff}"] = val

    # Log sample images
    for i, sample in enumerate(per_sample[:4]):
        item = val_dataset[selected[i]]
        generated = model.sample(
            ref1_img=item["ref1_img"].unsqueeze(0),
            ref2_img=item["ref2_img"].unsqueeze(0),
            pl_ref1=item["plucker_ref1"].unsqueeze(0),
            pl_ref2=item["plucker_ref2"].unsqueeze(0),
            pl_tgt=item["plucker_tgt"].unsqueeze(0),
            prompt=item.get("prompt", ""),
            num_steps=cfg.val_sample_steps,
            cfg_scale=cfg.val_cfg_scale,
            seed=cfg.seed,
        )[0].cpu()

        strip = np.concatenate([
            to_uint8(item["ref1_img"]),
            to_uint8(item["ref2_img"]),
            to_uint8(item["target_img"]),
            to_uint8(generated),
        ], axis=1)
        diff = item.get("difficulty", "?")
        caption = (f'{item["scene_name"]} | {diff} | '
                   f'PSNR={sample["psnr"]:.1f} LPIPS={sample.get("lpips", 0):.3f} '
                   f'CR={sample.get("copy_ratio", 0):.2f}')
        log_dict[f"val/sample_{i}"] = wandb.Image(strip, caption=caption)

    wandb.log(log_dict, step=global_step)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PoseSD with multi-GPU support")

    # Paths
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--scenes-file", type=str, default=None)
    p.add_argument("--output", type=str, default="checkpoints/pose_sd_v2")
    p.add_argument("--split-dir", type=str, default=None)
    p.add_argument("--resume-from", type=str, default=None)

    # Model
    p.add_argument("--pretrained-model", type=str, default="manojb/stable-diffusion-2-1-base")
    p.add_argument("--train-mode", choices=["cond", "full"], default="cond")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--xformers-attention", action="store_true")

    # Data — resolution
    p.add_argument("--H", type=int, default=DataConfig.H)
    p.add_argument("--W", type=int, default=DataConfig.W)

    # Data — triplet mining (defaults from DataConfig)
    p.add_argument("--max-triplets-per-scene", type=int, default=DataConfig.max_triplets_per_scene)
    p.add_argument("--min-points-per-image", type=int, default=DataConfig.min_points_per_image)
    p.add_argument("--min-orientation-dot", type=float, default=DataConfig.min_orientation_dot)
    p.add_argument("--max-focal-length-ratio", type=float, default=DataConfig.max_focal_length_ratio)
    p.add_argument("--min-ref-covisibility", type=float, default=DataConfig.min_ref_covisibility)
    p.add_argument("--max-ref-covisibility", type=float, default=DataConfig.max_ref_covisibility)
    p.add_argument("--near-duplicate-threshold", type=float, default=DataConfig.near_duplicate_threshold)
    p.add_argument("--no-reject-near-duplicates", action="store_true")
    p.add_argument("--max-pairs-per-target", type=int, default=6)
    p.add_argument("--pair-similarity-thresh", type=float, default=0.03)

    # Data — bucket covisibility/distance ranges
    p.add_argument("--easy-min-covis", type=float, default=DataConfig.easy_min_covis)
    p.add_argument("--easy-max-covis", type=float, default=DataConfig.easy_max_covis)
    p.add_argument("--easy-min-distance", type=float, default=DataConfig.easy_min_distance)
    p.add_argument("--easy-max-distance", type=float, default=DataConfig.easy_max_distance)
    p.add_argument("--medium-min-covis", type=float, default=DataConfig.medium_min_covis)
    p.add_argument("--medium-max-covis", type=float, default=DataConfig.medium_max_covis)
    p.add_argument("--medium-min-distance", type=float, default=DataConfig.medium_min_distance)
    p.add_argument("--medium-max-distance", type=float, default=DataConfig.medium_max_distance)
    p.add_argument("--hard-min-covis", type=float, default=DataConfig.hard_min_covis)
    p.add_argument("--hard-max-covis", type=float, default=DataConfig.hard_max_covis)
    p.add_argument("--hard-min-distance", type=float, default=DataConfig.hard_min_distance)
    p.add_argument("--hard-max-distance", type=float, default=DataConfig.hard_max_distance)

    # Training
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--per-gpu-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--lr-scheduler", choices=["cosine", "constant_with_warmup"], default="cosine")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--mixed-precision", choices=["bf16", "fp16", "no"], default="bf16")

    # EMA
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--no-ema", action="store_true")

    # Conditioning dropout
    p.add_argument("--cond-both-kept", type=float, default=0.85)
    p.add_argument("--cond-one-dropped", type=float, default=0.10)
    p.add_argument("--cond-both-dropped", type=float, default=0.05)

    # Slot randomization
    p.add_argument("--no-randomize-slots", action="store_true")

    # Bucket ratios
    p.add_argument("--easy-ratio", type=float, default=0.50)
    p.add_argument("--medium-ratio", type=float, default=0.35)
    p.add_argument("--hard-ratio", type=float, default=0.15)

    # Checkpoints & validation
    p.add_argument("--save-every-steps", type=int, default=10_000)
    p.add_argument("--val-every-steps", type=int, default=5_000)
    p.add_argument("--keep-checkpoints", type=int, default=5)
    p.add_argument("--val-sample-steps", type=int, default=50)
    p.add_argument("--val-cfg-scale", type=float, default=4.0)

    # Split
    p.add_argument("--test-scenes-pct", type=float, default=5.0)
    p.add_argument("--test-targets-per-scene", type=int, default=1)

    # W&B
    p.add_argument("--wandb-project", type=str, default="CoupledSceneSampling")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")

    return p.parse_args()


def _args_to_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_gpu_batch_size=args.per_gpu_batch_size,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler=args.lr_scheduler,
        ema_enabled=not args.no_ema,
        ema_decay=args.ema_decay,
        save_every_steps=args.save_every_steps,
        val_every_steps=args.val_every_steps,
        keep_checkpoints=args.keep_checkpoints,
        val_sample_steps=args.val_sample_steps,
        val_cfg_scale=args.val_cfg_scale,
        seed=args.seed,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_mode=args.wandb_mode,
        output_dir=args.output,
        scenes_file=args.scenes_file,
        scenes=args.scenes or [],
        split_dir=args.split_dir,
        resume_from=args.resume_from,
        randomize_slot_order=not args.no_randomize_slots,
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cnfg = _args_to_train_config(args)

    _setup_distributed()
    _set_seed(cnfg.seed)

    device = torch.device(f"cuda:{_local_rank()}")
    is_main = _is_main_process()

    output_dir = Path(cnfg.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # W&B init (main process only)
    if is_main and _WANDB_AVAILABLE and cnfg.wandb_mode != "disabled":
        wandb.init(
            project=cnfg.wandb_project, name=cnfg.wandb_name, mode=cnfg.wandb_mode,
            config=vars(args),
        )

    # Scenes
    scenes = list(dict.fromkeys((cnfg.scenes or []) + _read_lines(cnfg.scenes_file)))
    if not scenes:
        raise ValueError("Provide --scenes or --scenes-file")

    # Dataset — all mining params explicitly wired from CLI
    if is_main:
        print(f"Building dataset at {args.H}x{args.W}...")
    dataset = MegaScenesDataset(
        scene_dirs=scenes, H=args.H, W=args.W,
        # Bucket ranges
        easy_min_covis=args.easy_min_covis, easy_max_covis=args.easy_max_covis,
        easy_min_distance=args.easy_min_distance, easy_max_distance=args.easy_max_distance,
        medium_min_covis=args.medium_min_covis, medium_max_covis=args.medium_max_covis,
        medium_min_distance=args.medium_min_distance, medium_max_distance=args.medium_max_distance,
        hard_min_covis=args.hard_min_covis, hard_max_covis=args.hard_max_covis,
        hard_min_distance=args.hard_min_distance, hard_max_distance=args.hard_max_distance,
        # Bucket ratios
        easy_ratio=args.easy_ratio,
        medium_ratio=args.medium_ratio,
        hard_ratio=args.hard_ratio,
        # Ref-ref constraints
        min_ref_covisibility=args.min_ref_covisibility,
        max_ref_covisibility=args.max_ref_covisibility,
        # Quality filtering
        max_triplets_per_scene=args.max_triplets_per_scene,
        min_orientation_dot=args.min_orientation_dot,
        max_focal_length_ratio=args.max_focal_length_ratio,
        min_points_per_image=args.min_points_per_image,
        reject_near_duplicate_refs=not args.no_reject_near_duplicates,
        near_duplicate_threshold=args.near_duplicate_threshold,
        max_pairs_per_target=args.max_pairs_per_target,
        pair_similarity_thresh=args.pair_similarity_thresh,
    )

    # Split
    train_indices, test_indices, withheld_target_indices, split_info = _build_split(
        dataset.triplets, cnfg.seed, args.test_scenes_pct, args.test_targets_per_scene,
    )
    if is_main:
        print(f"Train: {len(train_indices)} | Test: {len(test_indices)} "
              f"({len(withheld_target_indices)} withheld-target)")

        split_dir = Path(cnfg.split_dir) if cnfg.split_dir else output_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Build weighted sampler for bucket ratios
    bucket_ratios = {
        Difficulty.EASY: args.easy_ratio,
        Difficulty.MEDIUM: args.medium_ratio,
        Difficulty.HARD: args.hard_ratio,
    }

    if dist.is_initialized():
        train_sampler = DistributedWeightedSampler(
            dataset, train_indices, bucket_ratios, seed=cnfg.seed,
        )
    else:
        train_sampler = _build_weighted_sampler(dataset, train_indices, bucket_ratios)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cnfg.per_gpu_batch_size,
        sampler=train_sampler,
        num_workers=cnfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = PoseSD(pretrained_model=args.pretrained_model, device=str(device))
    model.configure_trainable(args.train_mode)
    model.configure_memory_optimizations(
        gradient_checkpointing=args.gradient_checkpointing,
        xformers_attention=args.xformers_attention,
    )

    trainable_params = model.get_trainable_parameters()
    if is_main:
        print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
        eff_batch = cnfg.per_gpu_batch_size * _world_size() * cnfg.gradient_accumulation_steps
        print(f"Effective batch size: {eff_batch} "
              f"({cnfg.per_gpu_batch_size} x {_world_size()} GPUs x {cnfg.gradient_accumulation_steps} accum)")

    # DDP wrapping
    if dist.is_initialized():
        model.unet = DDP(model.unet, device_ids=[_local_rank()], find_unused_parameters=False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cnfg.lr,
        betas=cnfg.betas,
        weight_decay=cnfg.weight_decay,
        eps=cnfg.eps,
    )

    # LR scheduler
    lr_scheduler = _build_lr_scheduler(
        optimizer, cnfg.warmup_steps, cnfg.total_steps, cnfg.lr_scheduler,
    )

    # EMA
    ema = None
    if cnfg.ema_enabled:
        ema = EMAModel(trainable_params, decay=cnfg.ema_decay)

    # Resume
    global_step = 0
    start_epoch = 0
    if cnfg.resume_from:
        if is_main:
            print(f"Resuming from {cnfg.resume_from}")
        resumed = load_pose_sd_checkpoint(
            model, cnfg.resume_from, device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
        )
        global_step = resumed["global_step"]
        start_epoch = resumed["epoch"]

    # AMP scaler
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
    amp_dtype = dtype_map[cnfg.mixed_precision]
    use_amp = cnfg.mixed_precision != "no"

    # Training loop
    if is_main:
        print(f"\nStarting training for {cnfg.total_steps} steps...")

    epoch = start_epoch
    done = False

    try:
        while not done:
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            model.train()
            if is_main:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            else:
                pbar = train_loader

            accum_loss = 0.0
            accum_steps = 0
            bucket_losses: dict[str, list[float]] = {}

            for batch_idx, batch in enumerate(pbar):
                # Forward pass
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    loss, meta = model.training_step(
                        batch,
                        both_kept=args.cond_both_kept,
                        one_dropped=args.cond_one_dropped,
                        both_dropped=args.cond_both_dropped,
                        randomize_slots=cnfg.randomize_slot_order,
                    )
                    loss = loss / cnfg.gradient_accumulation_steps

                loss.backward()
                accum_loss += loss.item() * cnfg.gradient_accumulation_steps
                accum_steps += 1

                # Track per-bucket losses
                if "difficulty" in batch:
                    for diff_val in batch["difficulty"]:
                        bucket_losses.setdefault(diff_val, []).append(loss.item() * cnfg.gradient_accumulation_steps)

                # Optimizer step at accumulation boundary
                if accum_steps % cnfg.gradient_accumulation_steps == 0:
                    if cnfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, cnfg.grad_clip)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    if ema is not None:
                        ema.update(trainable_params)

                    global_step += 1
                    avg_loss = accum_loss / cnfg.gradient_accumulation_steps
                    accum_loss = 0.0

                    if is_main:
                        if isinstance(pbar, tqdm):
                            pbar.set_postfix(loss=f"{avg_loss:.4f}", step=global_step, lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

                        if _WANDB_AVAILABLE and cnfg.wandb_mode != "disabled":
                            log_dict = {
                                "train/loss": avg_loss,
                                "train/lr": lr_scheduler.get_last_lr()[0],
                                "train/step": global_step,
                                "train/epoch": epoch + 1,
                                "train/n_both_kept": meta.get("n_both_kept", 0),
                                "train/n_one_dropped": meta.get("n_one_dropped", 0),
                                "train/n_both_dropped": meta.get("n_both_dropped", 0),
                            }
                            # Log per-bucket losses
                            for diff_key, losses in bucket_losses.items():
                                if losses:
                                    log_dict[f"train/loss_{diff_key}"] = np.mean(losses)
                            wandb.log(log_dict, step=global_step)
                            bucket_losses.clear()

                    # Validation
                    if global_step % cnfg.val_every_steps == 0 and is_main:
                        if ema is not None:
                            ema.apply_shadow(trainable_params)
                        _log_validation(
                            model, val_dataset,
                            list(range(len(val_dataset))),
                            global_step, cnfg,
                        )
                        if ema is not None:
                            ema.restore(trainable_params)
                        model.train()

                    # Save checkpoint
                    if global_step % cnfg.save_every_steps == 0 and is_main:
                        if ema is not None:
                            ema.apply_shadow(trainable_params)
                        save_pose_sd_checkpoint(
                            model, output_dir / f"unet_step_{global_step}.pt",
                            optimizer=optimizer, lr_scheduler=lr_scheduler,
                            ema=ema, epoch=epoch + 1, global_step=global_step,
                        )
                        if ema is not None:
                            ema.restore(trainable_params)

                        # Always save latest
                        save_pose_sd_checkpoint(
                            model, output_dir / "unet_latest.pt",
                            optimizer=optimizer, lr_scheduler=lr_scheduler,
                            ema=ema, epoch=epoch + 1, global_step=global_step,
                        )
                        _cleanup_checkpoints(output_dir, cnfg.keep_checkpoints)
                        print(f"Saved checkpoint at step {global_step}")

                    if global_step >= cnfg.total_steps:
                        done = True
                        break

            epoch += 1

        # Final save
        if is_main:
            if ema is not None:
                ema.apply_shadow(trainable_params)
            save_pose_sd_checkpoint(
                model, output_dir / "unet_final.pt",
                optimizer=optimizer, lr_scheduler=lr_scheduler,
                ema=ema, epoch=epoch, global_step=global_step,
            )
            print(f"Training complete. Final step: {global_step}")

    finally:
        if is_main and _WANDB_AVAILABLE:
            wandb.finish()
        _cleanup_distributed()


def _cleanup_checkpoints(output_dir: Path, keep: int = 5) -> None:
    ckpts = sorted(output_dir.glob("unet_step_*.pt"), key=lambda p: p.stat().st_mtime)
    for p in ckpts[:-keep]:
        print(f"Removing old checkpoint: {p.name}")
        p.unlink()


if __name__ == "__main__":
    main()
