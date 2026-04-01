"""Validation utilities for PoseSD training.

Provides:
- PSNR, SSIM, LPIPS metrics
- Fixed validation subsets by difficulty bucket
- Reference-copying detection (LPIPS between generated and nearest ref)
- Bucket-wise loss and metric logging
- Multi-seed generation for more representative evaluation
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

try:
    import lpips as _lpips_module
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Image metrics
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR between two images in [-1, 1] range. Returns dB."""
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse < 1e-10:
        return 100.0
    # images in [-1, 1], so max pixel value = 1.0, range = 2.0
    return float(10.0 * np.log10(4.0 / mse))


def ssim(
    pred: torch.Tensor, target: torch.Tensor,
    window_size: int = 11, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2,
) -> float:
    """Structural similarity between two (C, H, W) images in [-1, 1]."""
    # Rescale to [0, 1]
    pred = (pred.float() + 1.0) / 2.0
    target = (target.float() + 1.0) / 2.0

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    channels = pred.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5, channels, pred.device, pred.dtype)

    mu1 = F.conv2d(pred, kernel, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, kernel, padding=window_size // 2, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=channels) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def _gaussian_kernel(size: int, sigma: float, channels: int,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
    return kernel_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)


class LPIPSMetric:
    """Lazy-loaded LPIPS metric (VGG backbone)."""

    def __init__(self):
        self._net = None

    def _ensure_loaded(self, device: torch.device) -> None:
        if self._net is None:
            if not _LPIPS_AVAILABLE:
                raise ImportError("lpips package required: pip install lpips")
            self._net = _lpips_module.LPIPS(net="vgg").to(device).eval()
            self._net.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """LPIPS distance between two (C, H, W) images in [-1, 1]."""
        device = pred.device
        self._ensure_loaded(device)
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        return float(self._net(pred, target).item())


# ---------------------------------------------------------------------------
# Reference-copying detection
# ---------------------------------------------------------------------------

def reference_copy_score(
    generated: torch.Tensor,
    ref1: torch.Tensor,
    ref2: torch.Tensor,
    target_gt: torch.Tensor,
    lpips_fn: LPIPSMetric | None = None,
) -> dict[str, float]:
    """Detect if generated image copies a reference instead of synthesizing the target.

    Returns:
        dict with:
        - lpips_to_target: LPIPS(generated, ground truth target) — lower = better
        - lpips_to_nearest_ref: min(LPIPS(generated, ref1), LPIPS(generated, ref2))
        - copy_ratio: lpips_to_nearest_ref / lpips_to_target
            < 1.0 means generated is closer to a ref than to GT (likely copying)
            > 1.0 means generated is closer to GT (good)
    """
    if lpips_fn is None:
        lpips_fn = LPIPSMetric()

    d_target = lpips_fn(generated, target_gt)
    d_ref1 = lpips_fn(generated, ref1)
    d_ref2 = lpips_fn(generated, ref2)
    d_nearest_ref = min(d_ref1, d_ref2)

    copy_ratio = d_nearest_ref / max(d_target, 1e-6)

    return {
        "lpips_to_target": d_target,
        "lpips_to_nearest_ref": d_nearest_ref,
        "lpips_to_ref1": d_ref1,
        "lpips_to_ref2": d_ref2,
        "copy_ratio": copy_ratio,
    }


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------

@dataclass
class ValMetrics:
    """Aggregated metrics for a validation run."""
    psnr_mean: float = 0.0
    ssim_mean: float = 0.0
    lpips_mean: float = 0.0
    copy_ratio_mean: float = 0.0
    count: int = 0

    # Per-bucket metrics
    bucket_psnr: dict[str, float] | None = None
    bucket_ssim: dict[str, float] | None = None
    bucket_lpips: dict[str, float] | None = None
    bucket_copy_ratio: dict[str, float] | None = None


@dataclass
class SeedResult:
    """Metrics and generated image for a single seed."""
    seed: int
    generated: torch.Tensor
    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = 0.0
    copy_ratio: float = 1.0


@torch.inference_mode()
def run_validation(
    model,
    val_dataset,
    indices: list[int],
    *,
    num_steps: int = 50,
    cfg_scale: float = 3.0,
    cfg_text: float = 3.0,
    seed: int = 42,
    seeds_per_sample: int = 3,
    max_samples: int = 16,
    compute_lpips: bool = True,
) -> tuple[ValMetrics, list[dict]]:
    """Run validation on selected indices with multiple seeds per sample.

    For each validation sample, generates ``seeds_per_sample`` images using
    seeds ``[seed, seed+1, ..., seed+seeds_per_sample-1]``.  Metrics are
    averaged across seeds for each sample, then across samples.

    Returns:
        (aggregated_metrics, list_of_per_sample_dicts)

        Each per-sample dict contains:
          - "psnr", "ssim", "lpips", "copy_ratio": averaged across seeds
          - "seeds": list[SeedResult] with per-seed images and metrics
          - "scene_name", "target_name", "difficulty": metadata
    """
    lpips_fn = LPIPSMetric() if compute_lpips else None

    per_sample: list[dict] = []
    bucket_metrics: dict[str, list[dict]] = {}

    indices_to_eval = indices[:max_samples]

    seeds_list = [seed + si for si in range(seeds_per_sample)]

    for idx in indices_to_eval:
        item = val_dataset[idx]
        target_gt = item["target_img"]

        # Batch all seeds into a single sample() call
        all_generated = model.sample(
            ref1_img=item["ref1_img"].unsqueeze(0),
            ref2_img=item["ref2_img"].unsqueeze(0),
            pl_ref1=item["plucker_ref1"].unsqueeze(0),
            pl_ref2=item["plucker_ref2"].unsqueeze(0),
            pl_tgt=item["plucker_tgt"].unsqueeze(0),
            prompt=item.get("prompt", ""),
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            cfg_text=cfg_text,
            seed=seeds_list if seeds_per_sample > 1 else seed,
        ).cpu()

        seed_results: list[SeedResult] = []
        for si, current_seed in enumerate(seeds_list):
            generated = all_generated[si]

            sr = SeedResult(seed=current_seed, generated=generated,
                            psnr=psnr(generated, target_gt),
                            ssim=ssim(generated, target_gt))

            if lpips_fn is not None:
                sr.lpips = lpips_fn(generated, target_gt)
                copy_info = reference_copy_score(
                    generated, item["ref1_img"], item["ref2_img"],
                    target_gt, lpips_fn,
                )
                sr.copy_ratio = copy_info["copy_ratio"]

            seed_results.append(sr)

        # Average metrics across seeds
        sample_entry = {
            "psnr": float(np.mean([s.psnr for s in seed_results])),
            "ssim": float(np.mean([s.ssim for s in seed_results])),
            "scene_name": item["scene_name"],
            "target_name": item.get("target_name", ""),
            "difficulty": item.get("difficulty", "unknown"),
            "seeds": seed_results,
            # Keep first seed's generated for backward compat
            "generated": seed_results[0].generated,
        }

        if compute_lpips:
            sample_entry["lpips"] = float(np.mean([s.lpips for s in seed_results]))
            sample_entry["copy_ratio"] = float(np.mean([s.copy_ratio for s in seed_results]))

        per_sample.append(sample_entry)

        diff = sample_entry["difficulty"]
        bucket_metrics.setdefault(diff, []).append(sample_entry)

    # Aggregate
    if not per_sample:
        return ValMetrics(), []

    agg = ValMetrics(
        psnr_mean=float(np.mean([s["psnr"] for s in per_sample])),
        ssim_mean=float(np.mean([s["ssim"] for s in per_sample])),
        count=len(per_sample),
    )

    if compute_lpips and "lpips" in per_sample[0]:
        agg.lpips_mean = float(np.mean([s["lpips"] for s in per_sample]))
        agg.copy_ratio_mean = float(np.mean([s.get("copy_ratio", 1.0) for s in per_sample]))

    # Per-bucket aggregation
    agg.bucket_psnr = {}
    agg.bucket_ssim = {}
    agg.bucket_lpips = {}
    agg.bucket_copy_ratio = {}
    for diff, samples in bucket_metrics.items():
        agg.bucket_psnr[diff] = float(np.mean([s["psnr"] for s in samples]))
        agg.bucket_ssim[diff] = float(np.mean([s["ssim"] for s in samples]))
        if compute_lpips and "lpips" in samples[0]:
            agg.bucket_lpips[diff] = float(np.mean([s["lpips"] for s in samples]))
            agg.bucket_copy_ratio[diff] = float(np.mean([s.get("copy_ratio", 1.0) for s in samples]))

    return agg, per_sample


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert [-1, 1] tensor (C, H, W) to uint8 numpy (H, W, C)."""
    return ((t.clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
