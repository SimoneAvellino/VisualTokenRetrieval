"""Soft-DTW: a differentiable, temporal-shift-tolerant sequence distance.

Point-wise MSE compares ``pred[t]`` to ``target[t]`` and so punishes a sequence
that is correct but shifted by a few tokens. Soft-DTW (Cuturi & Blondel, 2017)
instead finds the lowest-cost *monotonic* alignment between the two sequences and
sums the costs along it, with the hard ``min`` of dynamic time warping replaced by
a smooth ``soft-min`` so the whole thing is differentiable:

    soft_dtw_gamma(X, Y) = min^gamma_{A in alignments} <A, Delta(X, Y)>
    min^gamma(a_1, ..., a_k) = -gamma * log sum_i exp(-a_i / gamma)

where ``Delta_{ij}`` is the pairwise cost between ``X_i`` and ``Y_j``. As
``gamma -> 0`` this recovers hard DTW; larger ``gamma`` is smoother and more
shift-tolerant. Monotonic alignment is what makes soft-DTW preferable to a set
distance (Chamfer) here: visual tokens are time-ordered, and DTW preserves that
order while Chamfer discards it.

This is a compact pure-PyTorch implementation. The forward recursion is built
out of per-cell tensors (no in-place writes), so autograd differentiates it
directly. Sequence lengths here are small (``<= ~64``), so the ``O(N*M)``
Python loop is cheap and the autograd graph stays tiny.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_min3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """Smooth minimum of three ``[B]`` tensors: ``-gamma * logsumexp(-x/gamma)``."""
    x = torch.stack([a, b, c], dim=-1) / (-gamma)
    return -gamma * torch.logsumexp(x, dim=-1)


def _cost_matrix(x: torch.Tensor, y: torch.Tensor, dist: str) -> torch.Tensor:
    """Pairwise cost ``[B, N, M]`` between ``x [B,N,D]`` and ``y [B,M,D]``.

    ``cosine``  -> ``1 - cos_sim`` (scale-invariant, bounded in ``[0, 2]``);
    ``l2`` / ``sqeuclidean`` -> squared Euclidean distance.
    """
    if dist == "cosine":
        xn = F.normalize(x, dim=-1)
        yn = F.normalize(y, dim=-1)
        return 1.0 - torch.bmm(xn, yn.transpose(1, 2))
    # squared euclidean: |x|^2 + |y|^2 - 2 x.y
    x2 = (x * x).sum(-1, keepdim=True)            # [B, N, 1]
    y2 = (y * y).sum(-1, keepdim=True).transpose(1, 2)  # [B, 1, M]
    xy = torch.bmm(x, y.transpose(1, 2))          # [B, N, M]
    return (x2 + y2 - 2.0 * xy).clamp_min(0.0)


def _soft_dtw_from_cost(D: torch.Tensor, gamma: float) -> torch.Tensor:
    """Soft-DTW value ``[B]`` from a pairwise cost matrix ``D [B, N, M]``."""
    B, N, M = D.shape
    inf = D.new_full((B,), float("inf"))
    # R has a 0-padded border; R[i][j] is a [B] tensor.
    R = [[None] * (M + 1) for _ in range(N + 1)]
    R[0][0] = D.new_zeros(B)
    for i in range(1, N + 1):
        R[i][0] = inf
    for j in range(1, M + 1):
        R[0][j] = inf
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            r = _soft_min3(R[i - 1][j], R[i][j - 1], R[i - 1][j - 1], gamma)
            R[i][j] = D[:, i - 1, j - 1] + r
    return R[N][M]


class SoftDTW(nn.Module):
    """Differentiable soft-DTW distance between two batched sequences.

    Args:
        gamma:      soft-min temperature; smaller is closer to hard DTW.
        normalize:  if True, return the unbiased
                    ``sdtw(x, y) - 0.5 (sdtw(x, x) + sdtw(y, y))`` so the
                    self-distance offset is removed (can be negative).
        dist:       pairwise cost, ``"cosine"`` or ``"l2"``.
        length_norm: if True, divide the value by the alignment-path length
                    (``~max(N, M)``) so the loss scale is sequence-length
                    independent — important when ``decode_mode="length_aware"``
                    yields variable-length targets.
    """

    def __init__(
        self,
        gamma: float = 0.1,
        normalize: bool = True,
        dist: str = "cosine",
        length_norm: bool = True,
    ) -> None:
        super().__init__()
        if dist not in {"cosine", "l2", "sqeuclidean"}:
            raise ValueError(f"Unknown dist: {dist}")
        self.gamma = float(gamma)
        self.normalize = normalize
        self.dist = "l2" if dist == "sqeuclidean" else dist
        self.length_norm = length_norm

    def _pair(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d = _soft_dtw_from_cost(_cost_matrix(x, y, self.dist), self.gamma)
        if self.normalize:
            dxx = _soft_dtw_from_cost(_cost_matrix(x, x, self.dist), self.gamma)
            dyy = _soft_dtw_from_cost(_cost_matrix(y, y, self.dist), self.gamma)
            d = d - 0.5 * (dxx + dyy)
        if self.length_norm:
            d = d / float(max(x.shape[1], y.shape[1]))
        return d

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        y_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-sample soft-DTW ``[B]``.

        ``x [B, N, D]``, ``y [B, M, D]``. When ``x_lengths`` / ``y_lengths`` are
        given (``[B]`` int tensors), each sample is sliced to its true length and
        evaluated independently, so right-padding does not pollute the alignment.
        """
        if x_lengths is None and y_lengths is None:
            return self._pair(x, y)
        B = x.shape[0]
        xl = x_lengths if x_lengths is not None else x.new_full((B,), x.shape[1], dtype=torch.long)
        yl = y_lengths if y_lengths is not None else y.new_full((B,), y.shape[1], dtype=torch.long)
        out = []
        for b in range(B):
            n = int(xl[b].clamp(min=1))
            m = int(yl[b].clamp(min=1))
            out.append(self._pair(x[b : b + 1, :n], y[b : b + 1, :m]))
        return torch.cat(out, dim=0)
