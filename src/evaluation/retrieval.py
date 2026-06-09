"""Cross-sample retrieval evaluation.

``cos_token`` / ``cos_seq`` measure how close a prediction is to its own ground
truth, but in an embedding space where every visual token is highly self-similar
they saturate — even predicting the global mean scores ~0.7. They cannot tell
whether the model actually *discriminates* the queried moment from other moments.

Retrieval metrics answer the question that matters: given a prediction, is its
own GT the nearest among *all* candidates? Candidates are the GT moments of every
other sample in the eval split (cross-sample), so no extra negatives are needed.

Reported:
    - ``recall@1/5/10`` — fraction of predictions whose own GT is within the top-k
      most similar GTs.
    - ``mrr``           — mean reciprocal rank of the own GT.
    - ``median_rank``   — median rank of the own GT (1 = perfect).
    - ``chance_r1``     — ``1 / N``, the Recall@1 of random guessing (reference).

A model collapsed onto the mean lands near ``chance_r1``; a model that uses the
query to localise the moment scores well above it.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def _masked_mean(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over the time axis (dim=1) using a binary mask ``[B, T]``."""
    m = mask.unsqueeze(-1).to(t.dtype)
    return (t * m).sum(1) / m.sum(1).clamp_min(1e-6)


@torch.no_grad()
def collect_vectors(
    predict_fn: Callable[[Dict[str, object]], torch.Tensor],
    move_fn: Callable[[Dict[str, object]], Dict[str, object]],
    loader: DataLoader,
    max_batches: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Run ``predict_fn`` over the loader; return masked mean-pooled pred & GT.

    Args:
        predict_fn: maps a (device-moved) batch to ``pred [B, L, D]``.
        move_fn: moves a raw batch to device/dtype (``train._move``).
        loader: eval dataloader.
        max_batches: if > 0, stop early (dry-run).

    Returns:
        ``(pred_vecs [N, D], gt_vecs [N, D], annotation_ids)`` as float32 numpy.
    """
    preds: List[torch.Tensor] = []
    gts: List[torch.Tensor] = []
    ids: List[str] = []
    for bi, rb in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        b = move_fn(rb)
        pred = predict_fn(b).float()
        target = b["target_tokens"].float()  # type: ignore[union-attr]
        mask = b["target_mask"]               # type: ignore[index]
        if pred.shape[1] > target.shape[1]:
            pred = pred[:, : target.shape[1]]
        preds.append(_masked_mean(pred, mask).cpu())
        gts.append(_masked_mean(target, mask).cpu())
        for meta in rb["metadata"]:  # type: ignore[index]
            ids.append(str(meta.get("annotation_id", len(ids))))
    return (
        torch.cat(preds, 0).numpy(),
        torch.cat(gts, 0).numpy(),
        ids,
    )


def retrieval_scores(pred_vecs: np.ndarray, gt_vecs: np.ndarray) -> Dict[str, float]:
    """Cross-sample retrieval metrics from pooled pred & GT vectors.

    For each prediction ``i`` we rank every GT ``j`` by cosine similarity and look
    at the rank of the *own* GT (``j == i``). With ``N`` samples, ``chance_r1`` is
    ``1/N``.
    """
    n = pred_vecs.shape[0]
    if n < 2:
        return {"recall@1": float("nan"), "recall@5": float("nan"),
                "recall@10": float("nan"), "mrr": float("nan"),
                "median_rank": float("nan"), "chance_r1": float("nan"), "n": float(n)}

    p = torch.tensor(pred_vecs)
    g = torch.tensor(gt_vecs)
    p = torch.nn.functional.normalize(p, dim=-1)
    g = torch.nn.functional.normalize(g, dim=-1)
    sim = p @ g.t()                                   # [N, N] cosine

    # rank of the own GT for each prediction (1 = best).
    own = sim.diagonal().unsqueeze(1)                 # [N, 1]
    ranks = (sim > own).sum(dim=1) + 1                # how many GTs beat the own one
    ranks = ranks.numpy()

    def recall_at(k: int) -> float:
        return float((ranks <= k).mean())

    return {
        "recall@1": recall_at(1),
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "mrr": float((1.0 / ranks).mean()),
        "median_rank": float(np.median(ranks)),
        "chance_r1": float(1.0 / n),
        "n": float(n),
    }
