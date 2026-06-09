"""Per-model training/validation loss curves.

The orchestrator keeps each model's loss history **in memory only** and calls
:func:`plot_loss_curve` once at the end of that model's run to render a PNG.
The raw numbers are intentionally not persisted — only the image is kept, which
is all that is needed to show overfitting (train loss falling while validation
loss turns back up) on a slide.

Training loss is recorded per optimizer step (dense, smooth even for few
epochs); validation loss is recorded once per epoch and drawn as markers on the
same x-axis (global step) so the two align.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curve(
    history: Dict[str, List[Tuple[int, float]]],
    output_path: str,
    model_name: str,
    stopped_early: bool = False,
) -> None:
    """Render train (per-step) and val (per-epoch) loss curves to ``output_path``.

    Args:
        history: dict with keys ``"train_steps"`` (list of ``(global_step, loss)``)
            and ``"val_epochs"`` (list of ``(global_step, loss)``).
        output_path: PNG path to write.
        model_name: title label.
        stopped_early: annotate the early-stopping point if True.
    """
    train_steps: Sequence[Tuple[int, float]] = history.get("train_steps", [])
    val_epochs: Sequence[Tuple[int, float]] = history.get("val_epochs", [])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if train_steps:
        xs = [s for s, _ in train_steps]
        ys = [l for _, l in train_steps]
        ax.plot(xs, ys, color="#1f77b4", alpha=0.7, lw=1.2, label="train loss (per step)")

    if val_epochs:
        xs = [s for s, _ in val_epochs]
        ys = [l for _, l in val_epochs]
        ax.plot(xs, ys, color="#d62728", marker="o", lw=1.8, label="val loss (per epoch)")
        best_i = min(range(len(ys)), key=lambda i: ys[i])
        ax.axvline(xs[best_i], color="#2ca02c", linestyle="--", lw=1.2,
                   label=f"best val @ step {xs[best_i]}")
        if stopped_early and val_epochs:
            ax.scatter([xs[-1]], [ys[-1]], color="black", zorder=5, s=60,
                       label="early stop")

    ax.set_xlabel("global step (optimizer updates)")
    ax.set_ylabel("loss")
    ax.set_title(f"Training / validation loss — {model_name}")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve: {output_path}")
