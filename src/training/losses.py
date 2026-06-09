"""Loss functions for GT visual-token reconstruction.

The objective composes four complementary terms (each weight may be zero):

    - **MSE** (point-wise) — pins absolute magnitude; a weak anchor that keeps
      soft-DTW from drifting in scale.
    - **cosine** (per-token ``1 - cos_sim``) — pins direction.
    - **soft-DTW** — temporal-shift-tolerant alignment distance (see
      :mod:`src.training.soft_dtw`); the main reconstruction term once enabled,
      because point-wise MSE penalises correct-but-shifted sequences.
    - **temporal InfoNCE** — contrastive term whose negatives are *same-video,
      time-shifted* windows (hard negatives), forcing the model to delineate the
      exact action boundary instead of merely identifying the right video. Works
      at ``batch_size == 1`` because negatives are intra-sample, not in-batch.

All terms honour an optional ``target_mask`` so variable-length targets
(``decode_mode="length_aware"``) are scored only over real tokens.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .soft_dtw import SoftDTW


def _masked_mean_time(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean over the time axis (dim=1) honouring ``mask [B, T]`` (None -> plain mean)."""
    if mask is None:
        return x.mean(1)
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp_min(1e-6)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """MSE over valid tokens only."""
    if mask is None:
        return F.mse_loss(pred, target)
    m = mask.unsqueeze(-1).to(pred.dtype)
    se = ((pred - target) ** 2) * m
    return se.sum() / m.sum().clamp_min(1e-6) / pred.shape[-1]


def _masked_token_cosine(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean per-token cosine similarity over valid tokens."""
    cs = F.cosine_similarity(pred, target, dim=-1)  # [B, T]
    if mask is None:
        return cs.mean()
    m = mask.to(cs.dtype)
    return (cs * m).sum() / m.sum().clamp_min(1e-6)


def temporal_infonce(
    pred: torch.Tensor,
    pos: torch.Tensor,
    negs: torch.Tensor,
    temperature: float,
    pred_mask: Optional[torch.Tensor] = None,
    pos_mask: Optional[torch.Tensor] = None,
    neg_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample InfoNCE with same-video temporal hard negatives.

    Args:
        pred: ``[B, L, D]``  model output (anchor).
        pos:  ``[B, L, D]``  GT tokens (positive).
        negs: ``[B, K, L, D]`` time-shifted same-video windows (hard negatives).
        temperature: softmax temperature.
        *_mask: optional time masks (``[B, L]`` / ``[B, K, L]``) for padded tokens.

    Contrasts the anchor against ``{positive} U negatives`` after masked
    mean-pooling over time. Independent of batch size — negatives come from the
    sample's own video, not from other batch elements.
    """
    a = F.normalize(_masked_mean_time(pred, pred_mask), dim=-1)     # [B, D]
    p = F.normalize(_masked_mean_time(pos, pos_mask), dim=-1)       # [B, D]
    B, K = negs.shape[0], negs.shape[1]
    nm = neg_mask.reshape(B * K, negs.shape[2]) if neg_mask is not None else None
    n = _masked_mean_time(negs.reshape(B * K, negs.shape[2], negs.shape[3]), nm)
    n = F.normalize(n, dim=-1).reshape(B, K, -1)                   # [B, K, D]
    pos_logit = (a * p).sum(-1, keepdim=True)                      # [B, 1]
    neg_logit = torch.einsum("bd,bkd->bk", a, n)                   # [B, K]
    logits = torch.cat([pos_logit, neg_logit], dim=1) / temperature
    tgt = torch.zeros(B, dtype=torch.long, device=pred.device)
    return F.cross_entropy(logits, tgt)


def _masked_norm_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """SmoothL1 on per-token L2 norm difference — fixes scale collapse."""
    pred_n = pred.norm(dim=-1)    # [B, T]
    gt_n = target.norm(dim=-1)    # [B, T]
    diff = F.smooth_l1_loss(pred_n, gt_n, reduction="none")  # [B, T]
    if mask is None:
        return diff.mean()
    m = mask.to(diff.dtype)
    return (diff * m).sum() / m.sum().clamp_min(1e-6)


class ReconstructionLoss(nn.Module):
    """Combined reconstruction loss (MSE + cosine + norm + soft-DTW + temporal InfoNCE).

    ``loss = mse_weight    * MSE
           + cosine_weight * (1 - cos_sim_token)
           + norm_weight   * SmoothL1(||pred_t|| - ||gt_t||)
           + softdtw_weight * soft_dtw(pred, target)
           + infonce_weight * temporal_infonce(pred, target, negs)
           + length_weight  * SmoothL1(pred_len, target_len)``

    Terms with zero weight (or missing inputs) are skipped. ``softdtw_weight`` and
    ``infonce_weight`` (with hard negatives) are the Phase-1 levers; set them >0
    in the config to activate shift-tolerance and boundary-aware contrast.
    ``norm_weight`` fixes scale collapse: predictions learning direction (cosine)
    but landing in a deflated region of the embedding space.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        cosine_weight: float = 0.1,
        norm_weight: float = 0.0,
        softdtw_weight: float = 0.0,
        infonce_weight: float = 0.0,
        length_weight: float = 0.0,
        infonce_temperature: float = 0.07,
        softdtw_gamma: float = 0.1,
        softdtw_dist: str = "cosine",
        softdtw_normalize: bool = True,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.norm_weight = norm_weight
        self.softdtw_weight = softdtw_weight
        self.infonce_weight = infonce_weight
        self.length_weight = length_weight
        self.temperature = infonce_temperature
        self.sdtw = (
            SoftDTW(gamma=softdtw_gamma, normalize=softdtw_normalize, dist=softdtw_dist)
            if softdtw_weight > 0
            else None
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        pred_len: Optional[torch.Tensor] = None,
        negs: Optional[torch.Tensor] = None,
        neg_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # length_aware decode emits a fixed `max_out_len` slots; the (padded)
        # target is shorter. Align on the time axis — slots 0..L-1 are supervised
        # against the GT, the rest are dropped (and counted only via the length head).
        if pred.shape[1] > target.shape[1]:
            pred = pred[:, : target.shape[1]]
        mse = _masked_mse(pred, target, target_mask)
        cos_tok = _masked_token_cosine(pred, target, target_mask)
        cos_loss = 1.0 - cos_tok
        loss = self.mse_weight * mse + self.cosine_weight * cos_loss

        norm_loss = pred.new_tensor(0.0)
        if self.norm_weight > 0:
            norm_loss = _masked_norm_loss(pred, target, target_mask)
            loss = loss + self.norm_weight * norm_loss

        dtw = pred.new_tensor(0.0)
        if self.sdtw is not None:
            dtw = self.sdtw(pred, target, target_lengths, target_lengths).mean()
            loss = loss + self.softdtw_weight * dtw

        info_nce = pred.new_tensor(0.0)
        if self.infonce_weight > 0 and negs is not None and negs.shape[1] > 0:
            info_nce = temporal_infonce(
                pred, target, negs, self.temperature,
                pred_mask=target_mask, pos_mask=target_mask, neg_mask=neg_mask,
            )
            loss = loss + self.infonce_weight * info_nce

        length = pred.new_tensor(0.0)
        if self.length_weight > 0 and pred_len is not None and target_lengths is not None:
            length = F.smooth_l1_loss(pred_len, target_lengths.to(pred_len.dtype))
            loss = loss + self.length_weight * length

        with torch.no_grad():
            seq_cos = F.cosine_similarity(
                _masked_mean_time(pred, target_mask),
                _masked_mean_time(target, target_mask),
                dim=-1,
            ).mean()
            # per-token norm ratio ||pred|| / ||gt|| — diagnoses scale collapse.
            pn = pred.norm(dim=-1)
            gn = target.norm(dim=-1).clamp_min(1e-6)
            ratio = pn / gn
            if target_mask is None:
                norm_ratio = ratio.mean()
            else:
                rm = target_mask.to(ratio.dtype)
                norm_ratio = (ratio * rm).sum() / rm.sum().clamp_min(1e-6)

        return loss, {
            "mse": float(mse.detach()),
            "cos_token": float(cos_tok.detach()),
            "cos_seq": float(seq_cos.detach()),
            "cos_loss": float(cos_loss.detach()),
            "norm": float(norm_loss.detach()),
            "norm_ratio": float(norm_ratio.detach()),
            "soft_dtw": float(dtw.detach()),
            "info_nce": float(info_nce.detach()),
            "length": float(length.detach()),
        }
