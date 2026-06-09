"""Shared evaluation metrics for predicted vs. ground-truth visual tokens.

These metrics are computed identically for trained models and for the trivial
baselines so the two are directly comparable on the same scale. All functions
honour an optional ``mask`` ``[B, T]`` (1 = real token, 0 = pad) so that
variable-length targets are scored only over real tokens.

Reported metrics:
    - ``mse``        — mean squared error over valid tokens (scale-dependent).
    - ``cos_token``  — mean per-token cosine similarity (direction, fine-grained).
    - ``cos_seq``    — cosine similarity between masked mean-pooled sequences
      (the headline number: semantic alignment of the whole moment).
    - ``norm_ratio`` — mean ``||pred_t|| / ||gt_t||`` over valid tokens; diagnoses
      *scale collapse* (ideal = 1.0, < 1 means deflated predictions).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _masked_mean_time(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean over the time axis (dim=1) honouring ``mask [B, T]`` (None -> plain mean)."""
    if mask is None:
        return x.mean(1)
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp_min(1e-6)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """MSE over valid tokens only."""
    if mask is None:
        return F.mse_loss(pred, target)
    m = mask.unsqueeze(-1).to(pred.dtype)
    se = ((pred - target) ** 2) * m
    return se.sum() / m.sum().clamp_min(1e-6) / pred.shape[-1]


def masked_token_cosine(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean per-token cosine similarity over valid tokens."""
    cs = F.cosine_similarity(pred, target, dim=-1)  # [B, T]
    if mask is None:
        return cs.mean()
    m = mask.to(cs.dtype)
    return (cs * m).sum() / m.sum().clamp_min(1e-6)


def seq_cosine(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Cosine similarity between masked mean-pooled sequences, averaged over batch."""
    return F.cosine_similarity(
        _masked_mean_time(pred, mask), _masked_mean_time(target, mask), dim=-1
    ).mean()


def norm_ratio(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean ratio of per-token L2 norms ``||pred_t|| / ||gt_t||`` over valid tokens."""
    ratio = pred.norm(dim=-1) / target.norm(dim=-1).clamp_min(1e-6)  # [B, T]
    if mask is None:
        return ratio.mean()
    m = mask.to(ratio.dtype)
    return (ratio * m).sum() / m.sum().clamp_min(1e-6)


@torch.no_grad()
def token_metrics(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Compute all comparison metrics for one batch as plain floats.

    Args:
        pred:   ``[B, T, D]`` predicted tokens.
        target: ``[B, T, D]`` ground-truth tokens.
        mask:   optional ``[B, T]`` validity mask.

    Returns:
        Dict with keys ``mse``, ``cos_token``, ``cos_seq``, ``norm_ratio``.
    """
    # length-aware decoders may emit more slots than the (padded) target.
    if pred.shape[1] > target.shape[1]:
        pred = pred[:, : target.shape[1]]
    return {
        "mse": float(masked_mse(pred, target, mask).detach()),
        "cos_token": float(masked_token_cosine(pred, target, mask).detach()),
        "cos_seq": float(seq_cosine(pred, target, mask).detach()),
        "norm_ratio": float(norm_ratio(pred, target, mask).detach()),
    }


class MetricAccumulator:
    """Running mean of per-batch metric dicts (weighted by number of samples)."""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._n = 0

    def update(self, metrics: Dict[str, float], batch_size: int = 1) -> None:
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + v * batch_size
        self._n += batch_size

    def average(self) -> Dict[str, float]:
        n = max(1, self._n)
        return {k: v / n for k, v in self._sums.items()}
