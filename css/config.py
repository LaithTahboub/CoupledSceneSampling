"""
Training configuration for PoseSD.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    pretrained_model: str = "manojb/stable-diffusion-2-1-base"
    train_mode: str = "cond"  # "cond" (conv_in + attn only) or "full"
    prediction_target: str = "epsilon"
    gradient_checkpointing: bool = True
    xformers_attention: bool = False
    include_high_res_attention: bool = False


@dataclass
class DataConfig:
    """Triplet sampling and filtering configuration."""
    H: int = 256
    W: int = 256

    # --- Difficulty bucket covisibility ranges ---
    # Easy: very close views, high overlap
    easy_min_covis: float = 0.45
    easy_max_covis: float = 0.80
    easy_min_distance: float = 0.02
    easy_max_distance: float = 0.15

    # Medium: moderate baseline
    medium_min_covis: float = 0.25
    medium_max_covis: float = 0.50
    medium_min_distance: float = 0.08
    medium_max_distance: float = 0.40

    # Hard: larger baseline
    hard_min_covis: float = 0.12
    hard_max_covis: float = 0.30
    hard_min_distance: float = 0.15
    hard_max_distance: float = 1.0

    # Bucket sampling ratios (must sum to 1.0)
    easy_ratio: float = 0.50
    medium_ratio: float = 0.35
    hard_ratio: float = 0.15

    # Reference-reference constraints
    min_ref_covisibility: float = 0.10
    max_ref_covisibility: float = 0.70

    # Quality filtering
    max_triplets_per_scene: int = 140
    min_orientation_dot: float = 0.5  # cos(60°) — reject if viewing directions diverge too much
    max_focal_length_ratio: float = 2.0  # reject pairs with very different focal lengths
    min_points_per_image: int = 400  # reject images with too few COLMAP points
    reject_near_duplicate_refs: bool = True  # reject ref pairs with covis > 0.90
    near_duplicate_threshold: float = 0.81

    # Train/test split
    test_scenes_pct: float = 5.0
    test_targets_per_scene: int = 1


@dataclass
class CondDropoutConfig:
    """Conditioning dropout probabilities for geometry-only training."""
    # Probability distribution over reference configurations
    both_refs_kept: float = 0.85
    one_ref_dropped: float = 0.10
    both_refs_dropped: float = 0.05

    # Text conditioning dropout (independent of ref dropout)
    text_drop_prob: float = 0.10  # 10% chance of dropping caption to empty string


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Optimizer
    optimizer: str = "AdamW"
    lr: float = 1e-4  # for cond mode; use ~3e-5 for full UNet
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    eps: float = 1e-8

    # Gradient
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1

    # Batch size
    per_gpu_batch_size: int = 2
    # effective_batch_size = per_gpu_batch_size * num_gpus * gradient_accumulation_steps

    # Schedule
    total_steps: int = 200_000
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"  # "cosine" or "constant_with_warmup"

    # EMA
    ema_enabled: bool = True
    ema_decay: float = 0.9999

    # Checkpointing
    save_every_steps: int = 10_000
    keep_checkpoints: int = 5

    # Validation
    val_every_steps: int = 3_000
    val_sample_steps: int = 50
    val_cfg_scale: float = 3.0
    val_cfg_text: float = 3.0
    val_seeds_per_sample: int = 3

    # Misc
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: str = "bf16"  # "bf16", "fp16", or "no"

    # Multi-GPU
    dist_backend: str = "nccl"

    # Logging
    wandb_project: str = "CoupledSceneSampling"
    wandb_name: str | None = None
    wandb_mode: str = "online"

    # Paths
    output_dir: str = "checkpoints/pose_sd_vn"
    scenes_file: str | None = None
    scenes: list[str] = field(default_factory=list)
    split_dir: str | None = None
    resume_from: str | None = None

    # Training regime
    # Fraction of batches using auxiliary regime (1ref+2tgt or 2ref+2tgt)
    auxiliary_regime_prob: float = 0.0  # disabled by default for phase 1

    # Slot randomization
    randomize_slot_order: bool = True
