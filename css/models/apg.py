"""Adaptive Projected Guidance (APG) for diffusion sampling.

Reference:
Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models
(arXiv:2410.02416, ICLR 2025), Algorithm 1.
"""

import torch


class _MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = float(momentum)
        self.running_average: torch.Tensor | None = None

    def reset(self) -> None:
        self.running_average = None

    def update(self, update_value: torch.Tensor) -> torch.Tensor:
        if self.running_average is None:
            self.running_average = update_value
        else:
            self.running_average = update_value + self.momentum * self.running_average
        return self.running_average


class AdaptiveProjectedGuidance:
    """Stateful APG guidance on denoised predictions."""

    def __init__(
        self,
        guidance_scale: float,
        eta: float = 0.0,
        momentum: float = -0.5,
        norm_threshold: float = 0.0,
        eps: float = 1e-12,
    ):
        self.guidance_scale = float(guidance_scale)
        self.eta = float(eta)
        self.norm_threshold = float(norm_threshold)
        self.eps = float(eps)
        self._momentum_buffer = _MomentumBuffer(momentum) if momentum != 0 else None

    def reset(self) -> None:
        if self._momentum_buffer is not None:
            self._momentum_buffer.reset()

    @staticmethod
    def _project(v0: torch.Tensor, v1: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Split v0 into components parallel and orthogonal to v1."""
        dtype = v0.dtype
        v0_64 = v0.double()
        v1_64 = v1.double()
        reduce_dims = tuple(range(1, v0.ndim))

        denom = torch.linalg.vector_norm(v1_64, dim=reduce_dims, keepdim=True).clamp_min(eps)
        v1_unit = v1_64 / denom

        v0_parallel = (v0_64 * v1_unit).sum(dim=reduce_dims, keepdim=True) * v1_unit
        v0_orthogonal = v0_64 - v0_parallel
        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

    def _rescale_norm(self, diff: torch.Tensor) -> torch.Tensor:
        if self.norm_threshold <= 0:
            return diff
        reduce_dims = tuple(range(1, diff.ndim))
        diff_norm = torch.linalg.vector_norm(diff, dim=reduce_dims, keepdim=True).clamp_min(self.eps)
        scale = torch.clamp(self.norm_threshold / diff_norm, max=1.0)
        return diff * scale

    def guide(self, pred_cond: torch.Tensor, pred_uncond: torch.Tensor) -> torch.Tensor:
        if self.guidance_scale <= 1.0:
            return pred_cond

        diff = pred_cond - pred_uncond
        if self._momentum_buffer is not None:
            diff = self._momentum_buffer.update(diff)

        diff = self._rescale_norm(diff)
        diff_parallel, diff_orthogonal = self._project(diff, pred_cond, self.eps)
        update = diff_orthogonal + self.eta * diff_parallel
        return pred_cond + (self.guidance_scale - 1.0) * update
