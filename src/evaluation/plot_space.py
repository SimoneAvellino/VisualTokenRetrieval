"""2D embedding-space visualization of predicted vs ground-truth tokens.

For one model (or baseline), the masked mean-pooled prediction and GT of every
eval sample are projected jointly to 2D with PCA and scattered: GT as filled
circles, predictions as crosses, joined by faint arrows.

This is the fastest way to *see* mode collapse: if the model ignores the query
and predicts something close to the global mean, all the prediction crosses pile
up in one tight cluster while the GT circles spread out. A model that localises
the moment produces predictions that track their own GT.

The vectors are the same ones collected for retrieval evaluation, so this adds
no extra forward passes.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


def plot_pred_vs_gt(
    gt_vecs: np.ndarray,
    pred_vecs: np.ndarray,
    output_path: str,
    name: str,
    seed: int = 42,
) -> None:
    """Joint-PCA scatter of GT (●) vs prediction (×), connected by arrows.

    Also annotates the mean cosine similarity and the spread ratio
    (pred-cluster std / gt-cluster std): a value << 1 signals collapse.
    """
    if gt_vecs.shape[0] < 2:
        print(f"[plot_space] too few samples for {name}; skipping.")
        return

    combined = np.concatenate([gt_vecs, pred_vecs], axis=0)
    reducer = PCA(n_components=2, random_state=seed)
    red = reducer.fit_transform(combined)
    n = gt_vecs.shape[0]
    gt_2d, pred_2d = red[:n], red[n:]
    ev = reducer.explained_variance_ratio_

    cos = F.cosine_similarity(torch.tensor(gt_vecs), torch.tensor(pred_vecs), dim=-1)
    mean_cos = float(cos.mean())
    # collapse diagnostic: how tightly do predictions cluster vs the GTs?
    spread = float(pred_2d.std(0).mean() / (gt_2d.std(0).mean() + 1e-6))

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(gt_2d[:, 0], gt_2d[:, 1], c="#1f77b4", marker="o", s=45, alpha=0.8,
               zorder=3, label="GT")
    ax.scatter(pred_2d[:, 0], pred_2d[:, 1], c="#d62728", marker="x", s=45, alpha=0.8,
               zorder=3, label="Pred")
    for g, p in zip(gt_2d, pred_2d):
        ax.annotate("", xy=p, xytext=g,
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.25, lw=0.7))

    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="#1f77b4", linestyle="None", markersize=8, label="GT"),
        Line2D([0], [0], marker="x", color="#d62728", linestyle="None", markersize=8, label="Pred"),
    ], loc="upper right", fontsize=10)
    ax.set_title(
        f"Predicted vs GT tokens (mean-pooled, PCA) — {name}\n"
        f"N={n}  mean cos={mean_cos:.3f}  "
        f"pred/GT spread={spread:.2f} {'(collapse!)' if spread < 0.4 else ''}\n"
        f"PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%",
        fontsize=11,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA scatter: {output_path}  (spread={spread:.2f}, mean_cos={mean_cos:.3f})")
