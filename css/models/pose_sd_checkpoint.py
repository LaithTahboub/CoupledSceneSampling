from __future__ import annotations

import re
from pathlib import Path

import torch


class EMAModel:
    """Minimal EMA helper for a fixed parameter list."""

    def __init__(self, params, decay: float = 0.9999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow_params = [p.detach().clone() for p in params]
        self.collected_params: list[torch.Tensor] | None = None

    def update(self, params) -> None:
        with torch.no_grad():
            for shadow, p in zip(self.shadow_params, params):
                shadow.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, params) -> None:
        self.collected_params = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p, shadow in zip(params, self.shadow_params):
                p.copy_(shadow.to(device=p.device, dtype=p.dtype))

    def restore(self, params) -> None:
        if self.collected_params is None:
            return
        with torch.no_grad():
            for p, live in zip(params, self.collected_params):
                p.copy_(live.to(device=p.device, dtype=p.dtype))
        self.collected_params = None

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow_params": [t.detach().cpu() for t in self.shadow_params],
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        loaded = state.get("shadow_params")
        if loaded is None:
            return
        if len(loaded) != len(self.shadow_params):
            raise ValueError(
                f"EMA param length mismatch: checkpoint={len(loaded)} current={len(self.shadow_params)}"
            )
        for i, t in enumerate(loaded):
            self.shadow_params[i] = t.to(
                device=self.shadow_params[i].device,
                dtype=self.shadow_params[i].dtype,
            ).clone()


def _extract_unet_state(raw_ckpt):
    if isinstance(raw_ckpt, dict):
        if "unet" in raw_ckpt and isinstance(raw_ckpt["unet"], dict):
            return raw_ckpt["unet"]
        if "unet_state_dict" in raw_ckpt and isinstance(raw_ckpt["unet_state_dict"], dict):
            return raw_ckpt["unet_state_dict"]
        if "state_dict" in raw_ckpt and isinstance(raw_ckpt["state_dict"], dict):
            return raw_ckpt["state_dict"]
        if all(isinstance(v, torch.Tensor) for v in raw_ckpt.values()):
            return raw_ckpt
    raise ValueError("Unsupported checkpoint format; expected UNet state dict bundle")


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def _infer_epoch_step_from_name(path: Path) -> tuple[int, int]:
    epoch, step = 0, 0
    m_epoch = re.search(r"epoch[_]?(\d+)", path.stem)
    if m_epoch:
        epoch = int(m_epoch.group(1))
    m_step = re.search(r"step(\d+)", path.stem)
    if m_step:
        step = int(m_step.group(1))
    return epoch, step


def load_pose_sd_checkpoint(
    model,
    ckpt_path: str | Path,
    device,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler=None,
    ema: EMAModel | None = None,
    strict: bool = True,
) -> dict[str, int]:
    ckpt_path = Path(ckpt_path)
    raw = torch.load(ckpt_path, map_location=device)
    unet_state = _strip_module_prefix(_extract_unet_state(raw))
    model.unet.load_state_dict(unet_state, strict=strict)

    epoch, global_step = _infer_epoch_step_from_name(ckpt_path)
    if isinstance(raw, dict):
        epoch = int(raw.get("epoch", epoch))
        global_step = int(raw.get("global_step", global_step))
        opt_state = raw.get("optimizer", raw.get("optimizer_state_dict"))
        sched_state = raw.get("lr_scheduler", raw.get("lr_scheduler_state_dict"))
        ema_state = raw.get("ema", raw.get("ema_state"))
        if optimizer is not None and opt_state is not None:
            optimizer.load_state_dict(opt_state)
        if lr_scheduler is not None and sched_state is not None:
            lr_scheduler.load_state_dict(sched_state)
        if ema is not None and ema_state is not None:
            ema.load_state_dict(ema_state)

    return {"epoch": epoch, "global_step": global_step}


def save_pose_sd_checkpoint(
    model,
    ckpt_path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler=None,
    ema: EMAModel | None = None,
    epoch: int = 0,
    global_step: int = 0,
) -> None:
    payload = {
        "format_version": 1,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "unet": model.unet.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
    }
    torch.save(payload, Path(ckpt_path))
