"""Trivial, non-trained baselines for visual-token reconstruction.

These baselines have **no parameters** and are never trained — they exist to put
a floor under the learned models. If a trained model cannot beat them, it has
learned nothing useful. They are evaluated with exactly the same metrics
(:mod:`src.evaluation.metrics`) and on exactly the same data split as the models,
so the numbers are directly comparable on the improvement ladder.

Baselines:
    - ``mean_token``      — predict the (masked) mean of the context tokens,
      repeated for every output position. Ignores the query entirely; measures
      "how good is just predicting the average token of the video". Note the
      context is the *cropped* sequence the dataset feeds the model (capped at
      ``max_seq_len`` and biased toward the GT region).
    - ``full_video_mean`` — predict the mean of *every* token of the whole video,
      uncapped (bypasses the ``max_seq_len`` crop by reading the chunk index
      directly, once per video). Query-blind and identical for all annotations of
      a video; it also "sees" tokens after the query and the GT itself, so it is
      a leaky/optimistic floor — useful precisely to bound how much the global
      video average already resembles any queried moment.
    - ``center_window``   — return the centre window of the context, of the same
      length as the target. Returns *real* consecutive tokens but located by a
      fixed heuristic rather than by the query; measures the value of locating
      the moment via the question.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
from torch.utils.data import DataLoader

from ..datasets.reconstruction_dataset import build_collate_fn
from ..datasets.token_index import TokenChunkIndex
from ..models.text_encoder import build_clip_tokenizer
from ..training.train import TrainConfig, _move, build_splits, resolve_device
from .metrics import MetricAccumulator, token_metrics

BASELINE_NAMES = ("mean_token", "full_video_mean", "center_window")


def _mean_token_pred(batch: Dict[str, object]) -> torch.Tensor:
    """Predict the masked-mean context token, broadcast to the target length."""
    ctx = batch["context_tokens"]            # [B, T_ctx, D]
    ctx_mask = batch["context_mask"]         # [B, T_ctx]
    target = batch["target_tokens"]          # [B, L, D]
    m = ctx_mask.unsqueeze(-1).to(ctx.dtype)
    ctx_mean = (ctx * m).sum(1) / m.sum(1).clamp_min(1e-6)   # [B, D]
    L = target.shape[1]
    return ctx_mean.unsqueeze(1).expand(-1, L, -1).contiguous()


def _center_window_pred(batch: Dict[str, object]) -> torch.Tensor:
    """Return the centre window of the context, aligned to each target's length."""
    ctx = batch["context_tokens"]            # [B, T_ctx, D]
    ctx_mask = batch["context_mask"]         # [B, T_ctx]
    target = batch["target_tokens"]          # [B, L, D]
    target_mask = batch["target_mask"]       # [B, L]
    B, L, D = target.shape
    pred = torch.zeros_like(target)
    for i in range(B):
        n_ctx = int(ctx_mask[i].sum().item())
        if n_ctx == 0:
            continue
        valid_ctx = ctx[i, :n_ctx]                      # [n_ctx, D]
        l = max(1, int(target_mask[i].sum().item()))    # real target length
        if l >= n_ctx:
            window = valid_ctx                          # use all we have
        else:
            start = (n_ctx - l) // 2
            window = valid_ctx[start : start + l]        # [l, D]
        w = window.shape[0]
        pred[i, :w] = window[:L]
    return pred


def _make_full_video_mean_pred(
    chunk_index: TokenChunkIndex,
) -> Callable[[Dict[str, object]], torch.Tensor]:
    """Build a predictor returning the mean of the *whole* video's tokens.

    Bypasses the dataset's ``max_seq_len`` crop by reading every chunk of the
    video straight from the index (``load_range`` with no bounds), then mean-
    pooling. The per-video mean is computed once and cached — it is identical for
    every annotation of that video. Output is broadcast to each target's length.
    """
    cache: Dict[str, torch.Tensor] = {}

    def predict(batch: Dict[str, object]) -> torch.Tensor:
        target = batch["target_tokens"]                 # [B, L, D]
        meta = batch["metadata"]                         # list of dicts
        b, length, d = target.shape
        out = target.new_zeros(b, d)
        for i, m in enumerate(meta):                     # type: ignore[arg-type]
            vid = str(m["video_id"])                     # type: ignore[index]
            if vid not in cache:
                tok, _ = chunk_index.load_range(vid, None, None)
                cache[vid] = (
                    tok.mean(0) if tok.numel() > 0 else torch.zeros(d)
                )
            out[i] = cache[vid].to(device=out.device, dtype=out.dtype)
        return out.unsqueeze(1).expand(-1, length, -1).contiguous()

    return predict


_BASELINES: Dict[str, Callable[[Dict[str, object]], torch.Tensor]] = {
    "mean_token": _mean_token_pred,
    "center_window": _center_window_pred,
}


def _masked_mean(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(t.dtype)
    return (t * m).sum(1) / m.sum(1).clamp_min(1e-6)


@torch.no_grad()
def evaluate_baseline(
    name: str,
    cfg: TrainConfig,
    split: str = "test",
    max_batches: int = 0,
    pca_scatter_path: str = "",
) -> Dict[str, float]:
    """Evaluate one named baseline on the given split; return averaged metrics.

    Computes the same token metrics as the models (mse/cos/norm_ratio) *and* the
    cross-sample retrieval metrics (recall@1/5, mrr), and optionally writes a
    pred-vs-GT PCA scatter — all from a single pass, so baselines and models are
    directly comparable on every chart.

    Args:
        name: one of :data:`BASELINE_NAMES`.
        cfg: training config (defines data paths and the video-level split).
        split: ``"test"`` or ``"val"``.
        max_batches: if > 0, stop after this many batches (used by dry-run).
        pca_scatter_path: if set, write the PCA scatter PNG here.
    """
    if name not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline {name!r}. Available: {sorted(BASELINE_NAMES)}")

    device = resolve_device(cfg.device)
    dtype = torch.bfloat16 if (cfg.dtype == "bf16" and device.type == "cuda") else torch.float32

    _, vl_ds, ts_ds, ci = build_splits(cfg)
    ds = ts_ds if split == "test" else vl_ds
    if ds is None:
        raise ValueError(f"No data for split={split!r}.")

    # full_video_mean needs the chunk index (reads whole videos, uncapped);
    # the others are pure functions of the batch.
    predict = _make_full_video_mean_pred(ci) if name == "full_video_mean" else _BASELINES[name]

    tok = build_clip_tokenizer(cfg.clip_model)
    cfn = build_collate_fn(tok, cfg.max_question_len)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=cfn,
        pin_memory=(device.type == "cuda"),
    )

    acc = MetricAccumulator()
    pred_pool, gt_pool = [], []
    for bi, rb in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        b = _move(rb, device, dtype)
        pred = predict(b).float()
        target = b["target_tokens"].float()
        mask = b["target_mask"]
        acc.update(token_metrics(pred, target, mask), batch_size=target.shape[0])
        pred_pool.append(_masked_mean(pred, mask).cpu())
        gt_pool.append(_masked_mean(target, mask).cpu())

    metrics = acc.average()

    # cross-sample retrieval + optional PCA scatter
    import numpy as np

    from .retrieval import retrieval_scores

    pred_vecs = torch.cat(pred_pool, 0).numpy().astype(np.float32)
    gt_vecs = torch.cat(gt_pool, 0).numpy().astype(np.float32)
    retr = retrieval_scores(pred_vecs, gt_vecs)
    metrics.update({
        "recall@1": retr["recall@1"], "recall@5": retr["recall@5"],
        "mrr": retr["mrr"], "median_rank": retr["median_rank"],
        "chance_r1": retr["chance_r1"],
    })

    print(f"[baseline] {name} ({split}): " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    if pca_scatter_path:
        from .plot_space import plot_pred_vs_gt

        plot_pred_vs_gt(gt_vecs, pred_vecs, pca_scatter_path, name, seed=cfg.seed)
    return metrics
