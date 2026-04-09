"""Training script for RelightFlux (3-view, Plucker-conditioned, Flux.1-dev backbone).

Flow matching training with velocity prediction.  Mirrors train_relight_sd.py
but targets the Flux DiT architecture.

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
from PIL import Image as PILImage, ImageDraw
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from tqdm import tqdm

from css.config import DataConfig, TrainConfig
from css.data.MegaScenesDataset import Difficulty, MegaScenesDataset, SceneRecord
from css.models.EMA import CPUEMAModel, EMAModel
from css.models.relight_flux import RelightFlux
from css.train.validation import ValMetrics, grid_artifact_score, psnr, run_validation, ssim, to_uint8

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Distributed helpers (same as train_relight_sd.py)
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
    from datetime import timedelta
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=120))
    torch.cuda.set_device(_local_rank())


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _wandb_is_active() -> bool:
    return _WANDB_AVAILABLE and getattr(wandb, "run", None) is not None


# ---------------------------------------------------------------------------
# Utilities (same as train_relight_sd.py)
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
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _ema_live_weight(decay: float, step: int) -> float:
    step = max(0, int(step))
    decay = float(decay)
    return 1.0 - decay ** step


def _should_use_ema_for_eval(cfg: TrainConfig, global_step: int) -> bool:
    # Before the shadow has absorbed a meaningful fraction of live weights,
    # EMA previews are mostly just the initialization and can be badly
    # misleading for fast LoRA runs.
    return _ema_live_weight(cfg.ema_decay, global_step) >= 0.25


def _compute_bucket_weights(
    dataset: MegaScenesDataset,
    indices: list[int],
    bucket_ratios: dict[Difficulty, float],
) -> list[float]:
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
    weights = _compute_bucket_weights(dataset, indices, bucket_ratios)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(indices),
        replacement=True,
    )


class DistributedWeightedSampler(Sampler[int]):
    """Distributed sampler that respects bucket-ratio weights."""

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

        self.total_size = int(math.ceil(len(indices) / num_replicas)) * num_replicas
        self.num_samples = self.total_size // num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(len(self.all_indices), generator=g).tolist()

        while len(perm) < self.total_size:
            perm.append(perm[len(perm) % len(self.all_indices)])

        rank_perm = perm[self.rank::self.num_replicas]

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

        weights_t = torch.tensor(weights, dtype=torch.double)
        if weights_t.sum() < 1e-12:
            sample_order = torch.randperm(len(rank_perm), generator=g).tolist()
        else:
            sample_order = torch.multinomial(
                weights_t, num_samples=self.num_samples, replacement=True,
                generator=g,
            ).tolist()

        yield from (rank_perm[s] for s in sample_order)


# ---------------------------------------------------------------------------
# Train/test split (identical to train_relight_sd.py)
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
# Checkpoint save/load for RelightFlux
# ---------------------------------------------------------------------------

def _unwrap_transformer(model: RelightFlux):
    """Get the raw transformer module, stripping DDP / PeftModel wrappers."""
    t = model.transformer
    # Strip DDP
    if hasattr(t, "module"):
        t = t.module
    return t


def _is_lora_model(model: RelightFlux) -> bool:
    """Check if the transformer is wrapped with PEFT LoRA."""
    t = model.transformer
    if hasattr(t, "module"):
        t = t.module
    return hasattr(t, "peft_config")


def _group_trainable_named_params(model: RelightFlux) -> dict[str, dict[str, torch.nn.Parameter]]:
    groups = {"x_embedder": {}, "lora": {}, "other": {}}
    for name, param in _unwrap_transformer(model).named_parameters():
        if not param.requires_grad:
            continue
        if "x_embedder" in name:
            groups["x_embedder"][name] = param
        elif "lora_" in name:
            groups["lora"][name] = param
        else:
            groups["other"][name] = param
    return groups


def _snapshot_trainable_params(model: RelightFlux) -> dict[str, dict[str, torch.Tensor]]:
    groups = _group_trainable_named_params(model)
    return {
        group: {
            name: param.detach().float().cpu().clone()
            for name, param in named.items()
        }
        for group, named in groups.items()
    }


def _compute_param_update_stats(
    model: RelightFlux,
    snapshot: dict[str, dict[str, torch.Tensor]] | None,
) -> dict[str, float]:
    if snapshot is None:
        return {}

    stats: dict[str, float] = {}
    groups = _group_trainable_named_params(model)
    for group, named in groups.items():
        if not named:
            continue
        delta_sq = 0.0
        base_sq = 0.0
        current_sq = 0.0
        count = 0
        for name, param in named.items():
            current = param.detach().float().cpu()
            start = snapshot[group][name]
            diff = current - start
            delta_sq += float(diff.square().sum().item())
            base_sq += float(start.square().sum().item())
            current_sq += float(current.square().sum().item())
            count += current.numel()
        if count == 0:
            continue
        stats[f"param/{group}_delta_rms"] = math.sqrt(delta_sq / count)
        stats[f"param/{group}_base_rms"] = math.sqrt(base_sq / count)
        stats[f"param/{group}_current_rms"] = math.sqrt(current_sq / count)
        stats[f"param/{group}_delta_rel"] = math.sqrt(delta_sq / max(base_sq, 1e-12))
    return stats


def _compute_grad_stats(model: RelightFlux) -> dict[str, float]:
    stats: dict[str, float] = {}
    groups = _group_trainable_named_params(model)
    for group, named in groups.items():
        grad_sq = 0.0
        param_sq = 0.0
        count = 0
        with_grad = 0
        for _, param in named.items():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            cur = param.detach().float()
            grad_sq += float(grad.square().sum().item())
            param_sq += float(cur.square().sum().item())
            count += grad.numel()
            with_grad += 1
        if count == 0:
            stats[f"grad/{group}_rms"] = 0.0
            stats[f"grad/{group}_rel"] = 0.0
            stats[f"grad/{group}_params_with_grad"] = 0.0
            continue
        stats[f"grad/{group}_rms"] = math.sqrt(grad_sq / count)
        stats[f"grad/{group}_rel"] = math.sqrt(grad_sq / max(param_sq, 1e-12))
        stats[f"grad/{group}_params_with_grad"] = float(with_grad)
    return stats


def save_relight_flux_checkpoint(
    model: RelightFlux,
    ckpt_path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler=None,
    ema: EMAModel | CPUEMAModel | None = None,
    epoch: int = 0,
    global_step: int = 0,
) -> None:
    transformer = _unwrap_transformer(model)
    is_lora = _is_lora_model(model)

    if is_lora:
        # Save only LoRA adapter weights + x_embedder (much smaller)
        lora_state = {
            k: v.detach().cpu() for k, v in transformer.state_dict().items()
            if "lora_" in k or "x_embedder" in k
        }
        t_state = lora_state
    else:
        t_state = transformer.state_dict()

    payload = {
        "format_version": 1,
        "backbone": "flux",
        "is_lora": is_lora,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "transformer": t_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
    }
    torch.save(payload, Path(ckpt_path))


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state_dict.items():
        while k.startswith(("module.", "_orig_mod.")):
            if k.startswith("module."):
                k = k[len("module."):]
            elif k.startswith("_orig_mod."):
                k = k[len("_orig_mod."):]
        cleaned[k] = v
    return cleaned


def load_relight_flux_checkpoint(
    model: RelightFlux,
    ckpt_path: str | Path,
    device,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler=None,
    ema: EMAModel | CPUEMAModel | None = None,
    strict: bool = True,
) -> dict[str, int]:
    import re

    ckpt_path = Path(ckpt_path)
    raw = torch.load(ckpt_path, map_location=device)

    # Extract transformer state dict
    if isinstance(raw, dict):
        if "transformer" in raw:
            t_state = raw["transformer"]
        elif "state_dict" in raw:
            t_state = raw["state_dict"]
        else:
            t_state = raw
    else:
        t_state = raw

    t_state = _strip_module_prefix(t_state)

    is_lora_ckpt = isinstance(raw, dict) and raw.get("is_lora", False)
    if is_lora_ckpt:
        # LoRA checkpoint: load only the adapter + x_embedder weights (non-strict)
        _unwrap_transformer(model).load_state_dict(t_state, strict=False)
    else:
        _unwrap_transformer(model).load_state_dict(t_state, strict=strict)

    epoch, global_step = 0, 0
    if isinstance(raw, dict):
        epoch = int(raw.get("epoch", 0))
        global_step = int(raw.get("global_step", 0))
        if optimizer is not None and raw.get("optimizer") is not None:
            optimizer.load_state_dict(raw["optimizer"])
        if lr_scheduler is not None and raw.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(raw["lr_scheduler"])
        if ema is not None and raw.get("ema") is not None:
            ema.load_state_dict(raw["ema"])

    # Fallback: infer from filename
    if global_step == 0:
        m = re.search(r"step(\d+)", ckpt_path.stem)
        if m:
            global_step = int(m.group(1))

    return {"epoch": epoch, "global_step": global_step}


# ---------------------------------------------------------------------------
# Validation & logging
# ---------------------------------------------------------------------------

def _build_val_grid(
    item: dict,
    seed_results: list,
    to_uint8_fn,
) -> np.ndarray:
    """Build a multi-seed validation grid (same layout as SD version)."""
    from css.train.validation import SeedResult

    ref1 = to_uint8_fn(item["ref1_img"])
    ref2 = to_uint8_fn(item["ref2_img"])
    gt = to_uint8_fn(item["target_img"])
    H, W = ref1.shape[0], ref1.shape[1]

    label_h = 16

    def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
        strip = PILImage.new("RGB", (W, label_h), (30, 30, 30))
        draw = ImageDraw.Draw(strip)
        draw.text((4, 1), text, fill=(220, 220, 220))
        return np.concatenate([np.array(strip), img], axis=0)

    top_panels = [
        _label_panel(ref1, "ref1"),
        _label_panel(ref2, "ref2"),
        _label_panel(gt, "GT"),
    ]
    top_row = np.concatenate(top_panels, axis=1)

    bottom_panels = []
    for sr in seed_results:
        gen = to_uint8_fn(sr.generated)
        label = f"seed {sr.seed}  P={sr.psnr:.1f} L={sr.lpips:.3f}"
        bottom_panels.append(_label_panel(gen, label))
    bottom_row = np.concatenate(bottom_panels, axis=1)

    target_w = max(top_row.shape[1], bottom_row.shape[1])
    def _pad_w(arr: np.ndarray, w: int) -> np.ndarray:
        if arr.shape[1] < w:
            pad = np.zeros((arr.shape[0], w - arr.shape[1], 3), dtype=np.uint8)
            return np.concatenate([arr, pad], axis=1)
        return arr

    grid = np.concatenate([
        _pad_w(top_row, target_w),
        _pad_w(bottom_row, target_w),
    ], axis=0)

    prompt_text = item.get("caption", "") or item.get("prompt", "")
    if prompt_text:
        prompt_text = prompt_text.strip()
        if len(prompt_text) > 220:
            prompt_text = prompt_text[:217] + "..."
        text_h = 20
        canvas = PILImage.new("RGB", (grid.shape[1], grid.shape[0] + text_h), (0, 0, 0))
        canvas.paste(PILImage.fromarray(grid), (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, grid.shape[0] + 3), prompt_text, fill=(200, 200, 200))
        grid = np.array(canvas)

    return grid


def _sanitize_debug_name(text: str, limit: int = 80) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    cleaned = cleaned.strip("._")
    if not cleaned:
        cleaned = "sample"
    return cleaned[:limit]


def _tensor_stats(t: torch.Tensor) -> dict[str, float | list[int]]:
    t = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()),
    }


def _save_tensor_png(path: Path, image: torch.Tensor) -> None:
    PILImage.fromarray(to_uint8(image)).save(path)


def _dump_flux_val_debug(
    model: RelightFlux,
    item: dict,
    seed: int,
    sample_idx: int,
    global_step: int,
    cfg: TrainConfig,
    sample_metrics: dict,
    target_grid: dict[str, float],
) -> Path:
    debug_root = Path(cfg.output_dir) / "val_debug"
    debug_root.mkdir(parents=True, exist_ok=True)

    scene_name = _sanitize_debug_name(item.get("scene_name", "scene"))
    target_name = _sanitize_debug_name(item.get("target_name", f"sample_{sample_idx}"))
    sample_dir = debug_root / f"step_{global_step:07d}_sample_{sample_idx}_{scene_name}_{target_name}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    generated_dbg, sample_debug = model.sample(
        ref1_img=item["ref1_img"].unsqueeze(0),
        ref2_img=item["ref2_img"].unsqueeze(0),
        pl_ref1=item["plucker_ref1"].unsqueeze(0),
        pl_ref2=item["plucker_ref2"].unsqueeze(0),
        pl_tgt=item["plucker_tgt"].unsqueeze(0),
        prompt=item.get("prompt", ""),
        num_steps=cfg.val_sample_steps,
        cfg_scale=cfg.val_cfg_scale,
        cfg_text=cfg.val_cfg_text,
        seed=seed,
        return_debug=True,
    )
    generated_dbg = generated_dbg[0].cpu()

    target = item["target_img"]
    ref1 = item["ref1_img"]
    ref2 = item["ref2_img"]

    target_recon = model.decode_latent(
        model.encode_image(target.unsqueeze(0).to(model.device))
    )[0].cpu()
    ref1_recon = model.decode_latent(
        model.encode_image(ref1.unsqueeze(0).to(model.device))
    )[0].cpu()
    ref2_recon = model.decode_latent(
        model.encode_image(ref2.unsqueeze(0).to(model.device))
    )[0].cpu()

    comp = np.concatenate([
        to_uint8(ref1), to_uint8(ref2), to_uint8(target), to_uint8(generated_dbg),
    ], axis=1)
    PILImage.fromarray(comp).save(sample_dir / "comparison.png")

    vae_comp = np.concatenate([
        to_uint8(target), to_uint8(target_recon),
        to_uint8(ref1), to_uint8(ref1_recon),
        to_uint8(ref2), to_uint8(ref2_recon),
    ], axis=1)
    PILImage.fromarray(vae_comp).save(sample_dir / "vae_roundtrip.png")

    _save_tensor_png(sample_dir / "generated.png", generated_dbg)
    _save_tensor_png(sample_dir / "target.png", target)
    _save_tensor_png(sample_dir / "target_vae_recon.png", target_recon)

    step_entries = []
    for step_info in sample_debug.get("decoded_steps", []):
        decoded = step_info["decoded"]
        step_filename = (
            f"denoise_{step_info['step_index']:02d}_"
            f"t{int(round(step_info['timestep'])):04d}.png"
        )
        _save_tensor_png(sample_dir / step_filename, decoded)
        step_entries.append({
            "step_index": step_info["step_index"],
            "timestep": step_info["timestep"],
            "latent_min": step_info["latent_min"],
            "latent_max": step_info["latent_max"],
            "latent_mean": step_info["latent_mean"],
            "latent_std": step_info["latent_std"],
            "decoded_grid_ratio": grid_artifact_score(decoded)["grid_ratio"],
            "file": step_filename,
        })

    payload = {
        "global_step": int(global_step),
        "sample_index": int(sample_idx),
        "scene_name": item.get("scene_name", ""),
        "target_name": item.get("target_name", ""),
        "difficulty": item.get("difficulty", ""),
        "prompt": item.get("prompt", ""),
        "seed": int(seed),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "attention_backend": sample_debug.get("attention_backend", "unknown"),
        "pack_unpack_max_abs_error": sample_debug.get("pack_unpack_max_abs_error", None),
        "sample_metrics": {
            k: v for k, v in sample_metrics.items() if k not in {"generated", "seeds"}
        },
        "seed_metrics": [
            {
                "seed": int(sr.seed),
                "psnr": float(sr.psnr),
                "ssim": float(sr.ssim),
                "lpips": float(sr.lpips),
                "copy_ratio": float(sr.copy_ratio),
                "grid_ratio": float(getattr(sr, "grid_ratio", 0.0)),
            }
            for sr in sample_metrics.get("seeds", [])
        ],
        "target_grid": target_grid,
        "generated_grid": grid_artifact_score(generated_dbg),
        "target_vae_grid": grid_artifact_score(target_recon),
        "target_vae_psnr": psnr(target_recon, target),
        "target_vae_ssim": ssim(target_recon, target),
        "target_stats": _tensor_stats(target),
        "generated_stats": _tensor_stats(generated_dbg),
        "target_vae_stats": _tensor_stats(target_recon),
        "ref1_vae_stats": _tensor_stats(ref1_recon),
        "ref2_vae_stats": _tensor_stats(ref2_recon),
        "scheduler_timesteps": sample_debug.get("scheduler_timesteps", []),
        "decoded_steps": step_entries,
    }
    (sample_dir / "debug.json").write_text(json.dumps(payload, indent=2))

    print(
        f"  Flux debug bundle saved to {sample_dir} | "
        f"grid={sample_metrics.get('grid_ratio', 0.0):.2f} "
        f"target_grid={target_grid.get('grid_ratio', 0.0):.2f} "
        f"pack/unpack_err={sample_debug.get('pack_unpack_max_abs_error', 0.0):.2e} "
        f"attn={sample_debug.get('attention_backend', 'unknown')}"
    )
    return sample_dir


@torch.inference_mode()
def _log_validation(
    model: RelightFlux,
    val_dataset,
    val_indices: list[int],
    global_step: int,
    cfg: TrainConfig,
) -> None:
    if not _is_main_process():
        return

    raw_transformer = model.transformer
    if isinstance(raw_transformer, DDP):
        model.transformer = raw_transformer.module
    model.eval()

    try:
        max_val = min(8, len(val_indices))
        bucket_indices: dict[str, list[int]] = {}
        base_ds = val_dataset.dataset if hasattr(val_dataset, "dataset") else val_dataset
        for i in val_indices:
            raw_idx = val_dataset.indices[i] if hasattr(val_dataset, "indices") else i
            if hasattr(base_ds, "records"):
                diff = base_ds.records[raw_idx].difficulty.value
            else:
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
            cfg_text=cfg.val_cfg_text,
            seed=cfg.seed,
            seeds_per_sample=cfg.val_seeds_per_sample,
            max_samples=max_val,
            compute_lpips=True,
        )

        force_debug = os.environ.get("CSS_FLUX_VAL_DEBUG", "").strip().lower() not in {
            "", "0", "false", "no", "off",
        }
        debug_budget = max(0, int(os.environ.get("CSS_FLUX_VAL_DEBUG_MAX", "1")))
        debug_dumps = 0
        sample_log_entries = []
        sample_images = {}

        for i, sample in enumerate(per_sample[:4]):
            item = val_dataset[selected[i]]
            target_grid = grid_artifact_score(item["target_img"])
            suspicious_grid = sample.get("grid_ratio", 0.0) > max(
                1.35, target_grid["grid_ratio"] + 0.15,
            )

            sample_log_entries.append(
                f"{i}:{item['scene_name']}:{item.get('target_name', '')} "
                f"grid={sample.get('grid_ratio', 0.0):.2f} "
                f"target_grid={target_grid['grid_ratio']:.2f}"
            )

            diff = item.get("difficulty", "?")
            caption = (f'{item["scene_name"]} | {diff} | '
                       f'PSNR={sample["psnr"]:.1f} LPIPS={sample.get("lpips", 0):.3f} '
                       f'CR={sample.get("copy_ratio", 0):.2f} '
                       f'Grid={sample.get("grid_ratio", 0):.2f} '
                       f'(avg {cfg.val_seeds_per_sample} seeds)')
            if _wandb_is_active():
                grid = _build_val_grid(item, sample["seeds"], to_uint8)
                sample_images[f"val/sample_{i}"] = wandb.Image(grid, caption=caption)

            print(
                f"  Val sample {i}: scene={item['scene_name']} target={item.get('target_name', '')} "
                f"grid={sample.get('grid_ratio', 0.0):.2f} "
                f"target_grid={target_grid['grid_ratio']:.2f}"
            )
            if debug_dumps < debug_budget and (force_debug or suspicious_grid):
                _dump_flux_val_debug(
                    model=model,
                    item=item,
                    seed=sample["seeds"][0].seed,
                    sample_idx=i,
                    global_step=global_step,
                    cfg=cfg,
                    sample_metrics=sample,
                    target_grid=target_grid,
                )
                debug_dumps += 1

        print(
            f"  Val step {global_step}: PSNR={metrics.psnr_mean:.2f} SSIM={metrics.ssim_mean:.4f} "
            f"LPIPS={metrics.lpips_mean:.4f} CopyRatio={metrics.copy_ratio_mean:.3f} "
            f"GridRatio={metrics.grid_ratio_mean:.3f}"
        )
        if sample_log_entries:
            print("  " + " | ".join(sample_log_entries))

        if not _wandb_is_active():
            return

        log_dict = {
            "val/psnr": metrics.psnr_mean,
            "val/ssim": metrics.ssim_mean,
            "val/lpips": metrics.lpips_mean,
            "val/copy_ratio": metrics.copy_ratio_mean,
            "val/grid_ratio": metrics.grid_ratio_mean,
        }

        if metrics.bucket_psnr:
            for diff, val in metrics.bucket_psnr.items():
                log_dict[f"val/psnr_{diff}"] = val
        if metrics.bucket_lpips:
            for diff, val in metrics.bucket_lpips.items():
                log_dict[f"val/lpips_{diff}"] = val
        if metrics.bucket_copy_ratio:
            for diff, val in metrics.bucket_copy_ratio.items():
                log_dict[f"val/copy_ratio_{diff}"] = val
        if metrics.bucket_grid_ratio:
            for diff, val in metrics.bucket_grid_ratio.items():
                log_dict[f"val/grid_ratio_{diff}"] = val
        log_dict.update(sample_images)

        wandb.log(log_dict, step=global_step)

    finally:
        model.transformer = raw_transformer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RelightFlux with multi-GPU support")

    # Paths
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--scenes-file", type=str, default=None)
    p.add_argument("--output", type=str, default="checkpoints/relight_flux_v1")
    p.add_argument("--split-dir", type=str, default=None)
    p.add_argument("--resume-from", type=str, default=None)

    # Model
    p.add_argument("--pretrained-model", type=str, default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--train-mode", choices=["cond", "full", "lora"], default="full")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--compile-transformer", action="store_true",
                    help="Use torch.compile on the transformer")

    # LoRA
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.0)

    # Data — resolution (512 for training; 1024 OOMs on backward pass with 12B + 3-view attn)
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=512)

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
    p.add_argument("--min-targets-per-scene", type=int, default=1)
    p.add_argument("--identity-aug-prob", type=float, default=0.03)
    p.add_argument("--random-crop-prob", type=float, default=0.15)

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
    p.add_argument("--total-steps", type=int, default=60_000)
    p.add_argument("--per-gpu-batch-size", type=int, default=1)  # Flux is ~12B, smaller batch
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)  # Lower LR for large model
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
    p.add_argument("--text-drop-prob", type=float, default=0.10)

    # Captions
    p.add_argument("--caption-dir", type=str, default=None)

    # Slot randomization
    p.add_argument("--no-randomize-slots", action="store_true")

    # Bucket ratios
    p.add_argument("--easy-ratio", type=float, default=0.50)
    p.add_argument("--medium-ratio", type=float, default=0.35)
    p.add_argument("--hard-ratio", type=float, default=0.15)

    # Checkpoints & validation
    p.add_argument("--save-every-steps", type=int, default=500)
    p.add_argument("--val-every-steps", type=int, default=200)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--val-sample-steps", type=int, default=28)
    p.add_argument("--val-cfg-scale", type=float, default=3.0)
    p.add_argument("--val-cfg-text", type=float, default=3.0)
    p.add_argument("--val-seeds-per-sample", type=int, default=3)

    # Split
    p.add_argument("--test-scenes-pct", type=float, default=5.0)
    p.add_argument("--test-targets-per-scene", type=int, default=1)

    # W&B
    p.add_argument("--wandb-project", type=str, default="CoupledSceneSampling")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--wandb-init-timeout", type=int, default=300)

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
        val_cfg_text=args.val_cfg_text,
        val_seeds_per_sample=args.val_seeds_per_sample,
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


def _warn_if_low_flux_val_steps(args: argparse.Namespace, cfg: TrainConfig) -> None:
    model_name = (args.pretrained_model or "").lower()
    if "flux.1-dev" not in model_name:
        return
    if cfg.val_sample_steps >= 12:
        return

    print(
        "[train_relight_flux] WARNING: "
        f"--val-sample-steps={cfg.val_sample_steps} is very low for FLUX.1-dev. "
        "That often yields blocky latent-grid previews because the VAE is "
        "decoding an under-denoised latent rather than a fully sampled image. "
        "Use roughly 28 steps for representative validation images."
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

    if is_main:
        _warn_if_low_flux_val_steps(args, cnfg)

    output_dir = Path(cnfg.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
    amp_dtype = dtype_map[cnfg.mixed_precision]
    use_amp = cnfg.mixed_precision != "no"

    # W&B init: prefer the requested mode, but don't let network timeouts kill training.
    if is_main and _WANDB_AVAILABLE and cnfg.wandb_mode != "disabled":
        init_kwargs = dict(
            project=cnfg.wandb_project,
            name=cnfg.wandb_name,
            config=vars(args),
            settings=wandb.Settings(init_timeout=args.wandb_init_timeout),
        )
        try:
            wandb.init(mode=cnfg.wandb_mode, **init_kwargs)
        except Exception as exc:
            if cnfg.wandb_mode == "online":
                print(
                    "[train_relight_flux] W&B online init failed "
                    f"({type(exc).__name__}: {exc}). Falling back to offline mode."
                )
                try:
                    wandb.init(mode="offline", **init_kwargs)
                except Exception as offline_exc:
                    print(
                        "[train_relight_flux] W&B offline init also failed "
                        f"({type(offline_exc).__name__}: {offline_exc}). Continuing without W&B."
                    )
            else:
                print(
                    "[train_relight_flux] W&B init failed "
                    f"({type(exc).__name__}: {exc}). Continuing without W&B."
                )

    # Scenes
    scenes = list(dict.fromkeys((cnfg.scenes or []) + _read_lines(cnfg.scenes_file)))
    if not scenes:
        raise ValueError("Provide --scenes or --scenes-file")

    # Dataset
    if is_main:
        print(f"Building dataset at {args.H}x{args.W}...")
    dataset = MegaScenesDataset(
        scene_dirs=scenes, H=args.H, W=args.W,
        caption_dir=args.caption_dir,
        easy_min_covis=args.easy_min_covis, easy_max_covis=args.easy_max_covis,
        easy_min_distance=args.easy_min_distance, easy_max_distance=args.easy_max_distance,
        medium_min_covis=args.medium_min_covis, medium_max_covis=args.medium_max_covis,
        medium_min_distance=args.medium_min_distance, medium_max_distance=args.medium_max_distance,
        hard_min_covis=args.hard_min_covis, hard_max_covis=args.hard_max_covis,
        hard_min_distance=args.hard_min_distance, hard_max_distance=args.hard_max_distance,
        easy_ratio=args.easy_ratio,
        medium_ratio=args.medium_ratio,
        hard_ratio=args.hard_ratio,
        min_ref_covisibility=args.min_ref_covisibility,
        max_ref_covisibility=args.max_ref_covisibility,
        max_triplets_per_scene=args.max_triplets_per_scene,
        min_orientation_dot=args.min_orientation_dot,
        max_focal_length_ratio=args.max_focal_length_ratio,
        min_points_per_image=args.min_points_per_image,
        reject_near_duplicate_refs=not args.no_reject_near_duplicates,
        near_duplicate_threshold=args.near_duplicate_threshold,
        max_pairs_per_target=args.max_pairs_per_target,
        pair_similarity_thresh=args.pair_similarity_thresh,
        min_targets_per_scene=args.min_targets_per_scene,
        identity_aug_prob=args.identity_aug_prob,
        random_crop_prob=args.random_crop_prob,
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
        persistent_workers=cnfg.num_workers > 0,
    )

    # Model
    if is_main:
        print(f"Loading RelightFlux from {args.pretrained_model}...")
    model = RelightFlux(
        pretrained_model=args.pretrained_model,
        device=str(device),
        transformer_dtype=amp_dtype,
    )
    model.configure_trainable(
        args.train_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.configure_memory_optimizations(
        gradient_checkpointing=args.gradient_checkpointing,
        compile_transformer=args.compile_transformer,
    )

    trainable_params = model.get_trainable_parameters()
    trainable_snapshot = _snapshot_trainable_params(model) if is_main else None
    if is_main:
        total_params = sum(p.numel() for p in model.transformer.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        print(f"Total transformer params: {total_params:,}")
        print(f"Trainable params: {trainable_count:,}")
        eff_batch = cnfg.per_gpu_batch_size * _world_size() * cnfg.gradient_accumulation_steps
        print(f"Effective batch size: {eff_batch} "
              f"({cnfg.per_gpu_batch_size} x {_world_size()} GPUs x {cnfg.gradient_accumulation_steps} accum)")
        update_stats = _compute_param_update_stats(model, trainable_snapshot)
        if update_stats:
            pretty = ", ".join(
                f"{k.split('/', 1)[1]}={v:.3e}" for k, v in sorted(update_stats.items())
            )
            print(f"Initial parameter stats: {pretty}")

    # DDP wrapping (LoRA needs find_unused_parameters since base model params are frozen)
    if dist.is_initialized():
        model.transformer = DDP(
            model.transformer, device_ids=[_local_rank()],
            find_unused_parameters=(args.train_mode == "lora"),
        )

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

    # EMA — use GPU EMA for LoRA (small param count), CPU EMA for full fine-tune
    ema = None
    if cnfg.ema_enabled:
        if args.train_mode == "lora":
            ema = EMAModel(trainable_params, decay=cnfg.ema_decay)
        else:
            ema = CPUEMAModel(trainable_params, decay=cnfg.ema_decay)

    # Resume
    global_step = 0
    start_epoch = 0
    if cnfg.resume_from:
        if is_main:
            print(f"Resuming from {cnfg.resume_from}")
        resumed = load_relight_flux_checkpoint(
            model, cnfg.resume_from, device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
        )
        global_step = resumed["global_step"]
        start_epoch = resumed["epoch"]

    # Training loop
    if is_main:
        print(f"\nStarting Flux training for {cnfg.total_steps} steps...")

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
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    loss, meta = model.training_step(
                        batch,
                        both_kept=args.cond_both_kept,
                        one_dropped=args.cond_one_dropped,
                        both_dropped=args.cond_both_dropped,
                        text_drop_prob=args.text_drop_prob,
                        randomize_slots=cnfg.randomize_slot_order,
                    )
                    loss = loss / cnfg.gradient_accumulation_steps

                loss.backward()
                accum_loss += loss.item() * cnfg.gradient_accumulation_steps
                accum_steps += 1

                if "difficulty" in batch:
                    for diff_val in batch["difficulty"]:
                        bucket_losses.setdefault(diff_val, []).append(loss.item() * cnfg.gradient_accumulation_steps)

                if accum_steps % cnfg.gradient_accumulation_steps == 0:
                    grad_stats = _compute_grad_stats(model) if is_main else {}
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

                        if _wandb_is_active():
                            log_dict = {
                                "train/loss": avg_loss,
                                "train/lr": lr_scheduler.get_last_lr()[0],
                                "train/step": global_step,
                                "train/epoch": epoch + 1,
                                "train/n_both_kept": meta.get("n_both_kept", 0),
                                "train/n_one_dropped": meta.get("n_one_dropped", 0),
                                "train/n_both_dropped": meta.get("n_both_dropped", 0),
                                "train/n_text_dropped": meta.get("n_text_dropped", 0),
                            }
                            if ema is not None:
                                log_dict["train/ema_live_weight"] = _ema_live_weight(
                                    cnfg.ema_decay, global_step,
                                )
                            log_dict.update(grad_stats)
                            log_dict.update(_compute_param_update_stats(model, trainable_snapshot))
                            for diff_key, losses in bucket_losses.items():
                                if losses:
                                    log_dict[f"train/loss_{diff_key}"] = np.mean(losses)
                            wandb.log(log_dict, step=global_step)
                            bucket_losses.clear()
                        elif global_step <= 5 or global_step % max(1, cnfg.val_every_steps) == 0:
                            stats = {}
                            stats.update(grad_stats)
                            stats.update(_compute_param_update_stats(model, trainable_snapshot))
                            if stats:
                                pretty = ", ".join(
                                    f"{k.split('/', 1)[1] if '/' in k else k}={v:.3e}"
                                    for k, v in sorted(stats.items())
                                )
                                print(f"  Train stats step {global_step}: {pretty}")
                                lora_grad = stats.get("grad/lora_rms", 0.0)
                                xembed_grad = stats.get("grad/x_embedder_rms", 0.0)
                                lora_delta = stats.get("param/lora_delta_rms", 0.0)
                                xembed_delta = stats.get("param/x_embedder_delta_rms", 0.0)
                                if global_step <= 10 and max(lora_grad, xembed_grad) < 1e-12:
                                    print(
                                        "[train_relight_flux] WARNING: trainable gradients are "
                                        "effectively zero; LoRA/x_embedder may not be receiving "
                                        "updates."
                                    )
                                elif global_step >= 10 and max(lora_delta, xembed_delta) < 1e-8:
                                    print(
                                        "[train_relight_flux] WARNING: trainable parameters have "
                                        "barely moved from initialization."
                                    )

                    # Validation (barrier so other ranks don't race ahead during val)
                    if global_step % cnfg.val_every_steps == 0:
                        if dist.is_initialized():
                            dist.barrier()
                        if is_main:
                            use_ema_preview = ema is not None and _should_use_ema_for_eval(cnfg, global_step)
                            if ema is not None and not use_ema_preview:
                                live_weight = _ema_live_weight(cnfg.ema_decay, global_step)
                                print(
                                    "[train_relight_flux] Skipping EMA for validation at "
                                    f"step {global_step}: shadow has only absorbed "
                                    f"{100.0 * live_weight:.1f}% of live weights."
                                )
                            if use_ema_preview:
                                ema.apply_shadow(trainable_params)
                            _log_validation(
                                model, val_dataset,
                                list(range(len(val_dataset))),
                                global_step, cnfg,
                            )
                            if use_ema_preview:
                                ema.restore(trainable_params)
                        if dist.is_initialized():
                            dist.barrier()
                        model.train()

                    # Save checkpoint
                    if global_step % cnfg.save_every_steps == 0 and is_main:
                        use_ema_ckpt = ema is not None and _should_use_ema_for_eval(cnfg, global_step)
                        if ema is not None and not use_ema_ckpt:
                            live_weight = _ema_live_weight(cnfg.ema_decay, global_step)
                            print(
                                "[train_relight_flux] Saving raw weights at step "
                                f"{global_step}: EMA shadow has only absorbed "
                                f"{100.0 * live_weight:.1f}% of live weights."
                            )
                        if use_ema_ckpt:
                            ema.apply_shadow(trainable_params)
                        save_relight_flux_checkpoint(
                            model, output_dir / f"transformer_step_{global_step}.pt",
                            optimizer=optimizer, lr_scheduler=lr_scheduler,
                            ema=ema, epoch=epoch + 1, global_step=global_step,
                        )
                        if use_ema_ckpt:
                            ema.restore(trainable_params)

                        save_relight_flux_checkpoint(
                            model, output_dir / "transformer_latest.pt",
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
            use_ema_final = ema is not None and _should_use_ema_for_eval(cnfg, global_step)
            if use_ema_final:
                ema.apply_shadow(trainable_params)
            save_relight_flux_checkpoint(
                model, output_dir / "transformer_final.pt",
                optimizer=optimizer, lr_scheduler=lr_scheduler,
                ema=ema, epoch=epoch, global_step=global_step,
            )
            if use_ema_final:
                ema.restore(trainable_params)
            print(f"Training complete. Final step: {global_step}")

    finally:
        if is_main and _wandb_is_active():
            wandb.finish()
        _cleanup_distributed()


def _cleanup_checkpoints(output_dir: Path, keep: int = 3) -> None:
    ckpts = sorted(output_dir.glob("transformer_step_*.pt"), key=lambda p: p.stat().st_mtime)
    for p in ckpts[:-keep]:
        print(f"Removing old checkpoint: {p.name}")
        p.unlink()


if __name__ == "__main__":
    main()
