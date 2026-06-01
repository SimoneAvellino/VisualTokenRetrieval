"""Loss functions for GT visual-token reconstruction.

The training objective combines:
    - MSE between predicted and target tokens;
    - per-token cosine distance ``1 - cos_sim`` (weighted by ``cosine_weight``);
    - optional in-batch InfoNCE on sequence-mean vectors (weighted by
      ``infonce_weight``), active only when ``batch_size > 1``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float,
    infonce_weight: float,
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined reconstruction loss.

    ``loss = MSE
            + cosine_weight  * (1 - cos_sim_token)
            + infonce_weight * in_batch_InfoNCE(pred_seq, gt_seq)``

    In-batch InfoNCE only contributes when ``pred.shape[0] > 1`` (it needs other
    samples in the batch as negatives). Returns ``(loss, metrics)`` where
    ``metrics`` holds detached scalar diagnostics.
    """
    mse = F.mse_loss(pred, target)
    cos_tok = F.cosine_similarity(
        pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1]), dim=-1
    ).mean()
    cos_loss = 1.0 - cos_tok
    loss = mse + cosine_weight * cos_loss

    info_nce = pred.new_tensor(0.0)
    if infonce_weight > 0 and pred.shape[0] > 1:
        p = F.normalize(pred.mean(1), dim=-1)
        t = F.normalize(target.mean(1), dim=-1)
        logits = p @ t.t() / temperature
        info_nce = F.cross_entropy(
            logits, torch.arange(pred.shape[0], device=pred.device)
        )
        loss = loss + infonce_weight * info_nce

    with torch.no_grad():
        seq_cos = F.cosine_similarity(pred.mean(1), target.mean(1), dim=-1).mean()

    return loss, {
        "mse": float(mse.detach()),
        "cos_token": float(cos_tok.detach()),
        "cos_seq": float(seq_cos.detach()),
        "cos_loss": float(cos_loss.detach()),
        "info_nce": float(info_nce.detach()),
    }
