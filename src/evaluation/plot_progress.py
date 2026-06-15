"""Cross-model "improvement ladder" chart.

Reads the results registry written by the ladder orchestrator
(:mod:`src.utils.results_registry`) and draws one figure that tells the whole
story at a glance: every baseline and every model, in ladder order, with the
headline metric (``cos_seq``) as bars and ``norm_ratio`` as a secondary line.
This is the figure for the "the model gets progressively smarter" slide.

It is safe to re-run at any time: it simply re-reads the registry, so the chart
grows as more models finish. Run standalone::

    python -m src.evaluation.plot_progress --config experiments/configs/ladder_cluster.yaml
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..utils.results_registry import load_registry

_BASELINE_COLOR = "#9E9E9E"
_MODEL_COLOR = "#1f77b4"


def plot_progress(registry_path: str, output_path: str, primary: str = "cos_token") -> None:
    """Draw the ladder chart from a results registry.

    Args:
        registry_path: path to the ladder summary JSON.
        output_path: PNG path to write.
        primary: headline metric for the bars (default ``"cos_token"`` — more
            discriminative than ``cos_seq``, which mean-pools first and so
            inflates the trivial-baseline floor).
    """
    reg = load_registry(registry_path)
    entries: List[Dict[str, object]] = reg["entries"]  # type: ignore[assignment]
    if not entries:
        print(f"[plot_progress] registry empty: {registry_path}")
        return

    names = [str(e["name"]) for e in entries]
    metrics = [e["metrics"] for e in entries]  # type: ignore[index]
    primary_vals = [float(m.get(primary, float("nan"))) for m in metrics]  # type: ignore[union-attr]
    norm_vals = [float(m.get("norm_ratio", float("nan"))) for m in metrics]  # type: ignore[union-attr]
    colors = [_BASELINE_COLOR if e.get("kind") == "baseline" else _MODEL_COLOR for e in entries]

    x = list(range(len(names)))
    fig, ax1 = plt.subplots(figsize=(max(8, 1.4 * len(names)), 6))

    ax1.bar(x, primary_vals, color=colors, alpha=0.85, zorder=2)

    # y-limits with headroom so value labels never collide with the top/legend.
    finite = [v for v in primary_vals if v == v]  # drop NaN
    vmax = max(finite) if finite else 1.0
    vmin = min(finite) if finite else 0.0
    top = vmax + 0.18 * max(abs(vmax), 1e-3)
    bottom = min(0.0, vmin) - 0.04 * max(abs(vmin), 1e-3)
    ax1.set_ylim(bottom, top)
    pad = 0.012 * (top - bottom)
    for xi, v in zip(x, primary_vals):
        if v != v:  # NaN
            continue
        # label above positive bars, below negative ones — never inside the bar.
        va = "bottom" if v >= 0 else "top"
        offset = pad if v >= 0 else -pad
        ax1.text(xi, v + offset, f"{v:.3f}", ha="center", va=va, fontsize=9)

    ax1.set_ylabel(f"{primary} (higher = better)", color=_MODEL_COLOR)
    ax1.tick_params(axis="y", labelcolor=_MODEL_COLOR)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.grid(axis="y", alpha=0.2, zorder=0)

    # secondary axis: norm_ratio, ideal = 1.0
    ax2 = ax1.twinx()
    norm_line, = ax2.plot(
        x, norm_vals, color="#FF9800", marker="s", lw=1.6, zorder=3, label="norm_ratio"
    )
    ax2.axhline(1.0, color="#FF9800", linestyle=":", lw=1.0, alpha=0.7)
    ax2.set_ylabel("norm_ratio (ideal = 1.0)", color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    # single combined legend placed ABOVE the axes so it never covers any bar.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_BASELINE_COLOR, label="baseline (not trained)"),
        plt.Rectangle((0, 0), 1, 1, color=_MODEL_COLOR, label="trained model"),
        norm_line,
    ]
    ax1.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        fontsize=9,
        frameon=False,
    )

    fig.suptitle("Reconstruction quality across the model ladder", y=1.02, fontsize=13)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ladder chart: {output_path}")


def _load_entries(registry_path: str):
    """Return (entries, names, kinds) or (None, ...) if empty."""
    reg = load_registry(registry_path)
    entries = reg["entries"]  # type: ignore[index]
    if not entries:
        print(f"[plot] registry empty: {registry_path}")
        return None, None, None
    names = [str(e["name"]) for e in entries]
    kinds = [e.get("kind") for e in entries]
    return entries, names, kinds


def plot_retrieval(registry_path: str, output_path: str) -> None:
    """Recall@1 bars (+ MRR line) per model, with the random-chance reference.

    Retrieval is the metric that reveals real query discrimination: a model
    collapsed onto the mean sits at ``chance``, a model that localises the moment
    rises above it. ``cos_token`` cannot show this — it saturates.
    """
    entries, names, kinds = _load_entries(registry_path)
    if entries is None:
        return
    metrics = [e["metrics"] for e in entries]  # type: ignore[index]
    r1 = [float(m.get("recall@1", float("nan"))) for m in metrics]  # type: ignore[union-attr]
    r5 = [float(m.get("recall@5", float("nan"))) for m in metrics]  # type: ignore[union-attr]
    mrr = [float(m.get("mrr", float("nan"))) for m in metrics]  # type: ignore[union-attr]
    chance = next((float(m.get("chance_r1")) for m in metrics  # type: ignore[union-attr]
                   if m.get("chance_r1") is not None), None)
    # Recall@1 is colored by kind (baseline vs model); Recall@5 is a lighter
    # companion bar next to it so both are visible per model.
    c1 = [_BASELINE_COLOR if k == "baseline" else _MODEL_COLOR for k in kinds]

    import numpy as np

    x = np.arange(len(names))
    w = 0.4
    fig, ax1 = plt.subplots(figsize=(max(8, 1.4 * len(names)), 6))
    ax1.bar(x - w / 2, r1, w, color=c1, alpha=0.9, zorder=2)
    ax1.bar(x + w / 2, r5, w, color=c1, alpha=0.45, zorder=2, hatch="//", edgecolor="white")
    for xi, v in zip(x, r1):
        if v == v:
            ax1.text(xi - w / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x, r5):
        if v == v:
            ax1.text(xi + w / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    if chance is not None:
        ax1.axhline(chance, color="black", ls="--", lw=1.2, label=f"chance (1/N = {chance:.3f})")
    ax1.set_ylabel("Recall (higher = better)", color=_MODEL_COLOR)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.grid(axis="y", alpha=0.2, zorder=0)

    ax2 = ax1.twinx()
    mrr_line, = ax2.plot(x, mrr, color="#FF9800", marker="s", lw=1.6, zorder=3, label="MRR")
    ax2.set_ylabel("MRR", color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_MODEL_COLOR, label="Recall@1"),
        plt.Rectangle((0, 0), 1, 1, color=_MODEL_COLOR, alpha=0.45, hatch="//", label="Recall@5"),
        plt.Rectangle((0, 0), 1, 1, color=_BASELINE_COLOR, label="baseline"),
        mrr_line,
    ]
    if chance is not None:
        handles.append(Line2D([0], [0], color="black", ls="--", label=f"chance = {chance:.3f}"))
    ax1.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
               ncol=5, fontsize=9, frameon=False)
    fig.suptitle("Retrieval: does the model rank its own moment first?", y=1.02, fontsize=13)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved retrieval chart: {output_path}")


def plot_metric_comparison(registry_path: str, output_path: str) -> None:
    """Grouped bars: cos_token (saturates) vs Recall@1 (discriminates), per model.

    This is the "reveal": cos_token is flat near the trivial-baseline floor for
    every model, while Recall@1 — read against the chance line — shows which
    models actually localise the moment.
    """
    entries, names, kinds = _load_entries(registry_path)
    if entries is None:
        return
    metrics = [e["metrics"] for e in entries]  # type: ignore[index]
    cos = [float(m.get("cos_token", float("nan"))) for m in metrics]  # type: ignore[union-attr]
    r1 = [float(m.get("recall@1", float("nan"))) for m in metrics]  # type: ignore[union-attr]
    chance = next((float(m.get("chance_r1")) for m in metrics  # type: ignore[union-attr]
                   if m.get("chance_r1") is not None), None)

    import numpy as np

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(names)), 6))
    ax.bar(x - w / 2, cos, w, color="#90A4AE", label="cos_token (saturates)")
    ax.bar(x + w / 2, r1, w, color="#1f77b4", label="Recall@1 (discriminates)")
    for xi, v in zip(x, cos):
        if v == v:
            ax.text(xi - w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x, r1):
        if v == v:
            ax.text(xi + w / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    if chance is not None:
        ax.axhline(chance, color="black", ls="--", lw=1.2, label=f"Recall@1 chance = {chance:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("score")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("cos_token vs Recall@1 — the saturating metric vs the honest one")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metric-comparison chart: {output_path}")


def main() -> None:
    """Standalone entrypoint: read a ladder config and redraw the chart."""
    from ..utils.config import load_yaml, parse_config_arg

    config_path = parse_config_arg("Plot the cross-model improvement ladder.")
    cfg = load_yaml(config_path)
    results_dir = cfg.get("results_dir", "./experiments/results")
    figures_dir = cfg.get("figures_dir", "./figures")
    summary = cfg.get("summary_name", "ladder_summary.json")
    reg_path = os.path.join(results_dir, summary)
    plot_progress(reg_path, os.path.join(figures_dir, "ladder_progress.png"))
    plot_retrieval(reg_path, os.path.join(figures_dir, "retrieval_progress.png"))
    plot_metric_comparison(reg_path, os.path.join(figures_dir, "metric_comparison.png"))


if __name__ == "__main__":
    main()
