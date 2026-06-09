"""Sequential "model ladder" orchestrator.

Runs, in one process and in a fixed order, the trivial baselines and then a
sequence of models whose loss function grows progressively more complex. The
goal is a single command that produces the whole "the model gets smarter"
story, plus the figures for it.

Pipeline::

    baselines (eval only)  ->  models (trained, increasing loss complexity)
                                          |
                       per-model loss curve PNG  +  one summary JSON entry
                                          |
                              cross-model ladder chart PNG

**Resume.** Every finished baseline/model writes one entry to the results
registry (:mod:`src.utils.results_registry`). On relaunch, any step already in
the registry is skipped, so an interrupted run (e.g. cluster time limit)
continues from the first unfinished step. Model weights are *not* needed for
resume and are deleted after each model's evaluation — only scores are kept.

**Dry run.** ``dry_run: true`` caps epochs and batches so the entire pipeline
(baselines, every model, both plot types) executes end-to-end in minutes,
verifying the wiring before a real run.

Run::

    python -m src.training.ladder --config experiments/configs/ladder_cluster.yaml
"""

from __future__ import annotations

import copy
import gc
import os
from dataclasses import dataclass, field
from typing import Dict, List

import torch

from ..evaluation.baselines import evaluate_baseline
from ..evaluation.plot_progress import (
    plot_metric_comparison,
    plot_progress,
    plot_retrieval,
)
from ..utils.config import from_dict, load_yaml, parse_config_arg
from ..utils.results_registry import append_entry, has_entry, make_entry
from .train import TrainConfig, run as train_run


@dataclass
class LadderConfig:
    """Configuration for a full ladder run."""

    base_config: str                                   # path to a base train YAML
    results_dir: str = "./experiments/results"
    figures_dir: str = "./figures"
    summary_name: str = "ladder_summary.json"
    eval_split: str = "test"                           # test | val
    early_stopping_patience: int = 5
    baselines: List[str] = field(default_factory=list)
    steps: List[Dict[str, object]] = field(default_factory=list)
    # dry-run: tiny end-to-end pass to validate the whole pipeline.
    dry_run: bool = False
    dry_run_epochs: int = 1
    dry_run_max_batches: int = 4


def _summary_path(cfg: LadderConfig) -> str:
    return os.path.join(cfg.results_dir, cfg.summary_name)


def _free_gpu() -> None:
    """Release the CUDA caching allocator between steps.

    Every baseline/model runs in this one process, so without an explicit
    release the caching allocator keeps each step's freed memory cached and the
    GPU footprint grows step after step until it exceeds the job's quota.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _build_train_cfg(
    base: Dict[str, object], lcfg: LadderConfig, name: str, overrides: Dict[str, object]
) -> TrainConfig:
    """Merge base config + per-step overrides + ladder-injected fields."""
    merged = copy.deepcopy(base)
    merged.update(overrides)
    base_ckpt = str(base.get("checkpoint_dir", "./experiments/checkpoints"))
    merged.update(
        {
            "run_name": name,
            "keep_checkpoints": False,                 # ladder needs scores, not weights
            "checkpoint_dir": os.path.join(base_ckpt, name),
            "loss_curve_path": os.path.join(lcfg.figures_dir, f"{name}_loss_curve.png"),
            "pca_scatter_path": os.path.join(lcfg.figures_dir, f"{name}_pca.png"),
            "early_stopping_patience": lcfg.early_stopping_patience,
        }
    )
    if lcfg.dry_run:
        merged.update(
            {
                "epochs": lcfg.dry_run_epochs,
                "max_train_batches": lcfg.dry_run_max_batches,
                "max_eval_batches": lcfg.dry_run_max_batches,
                "early_stopping_patience": 0,          # don't stop early in a tiny run
                "length_warmup_epochs": 0,
            }
        )
    return from_dict(TrainConfig, merged)


def run(lcfg: LadderConfig) -> None:
    """Execute the full ladder: baselines, models, then the ladder chart."""
    os.makedirs(lcfg.results_dir, exist_ok=True)
    os.makedirs(lcfg.figures_dir, exist_ok=True)
    summary = _summary_path(lcfg)
    base = load_yaml(lcfg.base_config)
    max_eval = lcfg.dry_run_max_batches if lcfg.dry_run else 0

    print(f"\n{'=' * 60}\nLADDER RUN — summary: {summary}\n{'=' * 60}")
    if lcfg.dry_run:
        print("** DRY RUN ** (capped epochs/batches; validates the pipeline)\n")

    # 1) trivial baselines (no training)
    base_cfg = from_dict(TrainConfig, base)
    for name in lcfg.baselines:
        if has_entry(summary, name):
            print(f"[skip] baseline {name} already done.")
            continue
        print(f"\n--- baseline: {name} ---")
        metrics = evaluate_baseline(
            name, base_cfg, split=lcfg.eval_split, max_batches=max_eval,
            pca_scatter_path=os.path.join(lcfg.figures_dir, f"{name}_pca.png"),
        )
        append_entry(summary, make_entry(name, "baseline", lcfg.eval_split, metrics))
        _free_gpu()

    # 2) trained models, increasing loss complexity
    for step in lcfg.steps:
        name = str(step["name"])
        overrides = dict(step.get("overrides", {}))  # type: ignore[arg-type]
        if has_entry(summary, name):
            print(f"[skip] model {name} already done.")
            continue
        print(f"\n--- model: {name} ---")
        tcfg = _build_train_cfg(base, lcfg, name, overrides)
        result = train_run(tcfg)
        _free_gpu()
        if result is None:
            print(f"[warn] {name} returned no result (dry-run of train?); skipping entry.")
            continue
        append_entry(
            summary,
            make_entry(
                name,
                "model",
                str(result["split"]),
                result["metrics"],  # type: ignore[arg-type]
                epochs_run=int(result.get("epochs_run", 0)),  # type: ignore[union-attr]
                stopped_early=bool(result.get("stopped_early", False)),  # type: ignore[union-attr]
            ),
        )

    # 3) cross-model charts
    plot_progress(summary, os.path.join(lcfg.figures_dir, "ladder_progress.png"))
    plot_retrieval(summary, os.path.join(lcfg.figures_dir, "retrieval_progress.png"))
    plot_metric_comparison(summary, os.path.join(lcfg.figures_dir, "metric_comparison.png"))
    print(f"\n{'=' * 60}\nLADDER DONE.\n{'=' * 60}")


def main() -> None:
    config_path = parse_config_arg("Run the full baseline + model ladder.")
    lcfg = from_dict(LadderConfig, load_yaml(config_path))
    run(lcfg)


if __name__ == "__main__":
    main()
