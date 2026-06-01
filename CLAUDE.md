# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is the **template/scaffold** for a deep learning course project (Furnari's course). Two distinct concerns live here:

1. **The project catalog** — `INSTRUCTIONS.md` lists ~32 available project tracks (ID, title, module, dataset, assigned group) plus a detailed spec section per project. Recent git history is almost entirely catalog maintenance (adding/editing project entries). When asked to "add project N" or "update project N", edit `INSTRUCTIONS.md`: keep the summary row in the `## Project List` table in sync with the detailed `<a name="project-N">` section below.
2. **The student scaffold** — `README.md`, `docs/REPORT.md`, `CONTRIBUTING.md`, `environment.yml`, `LICENSE`, and the `src/ data/ experiments/ notebooks/ figures/` tree. These are templates a student fork fills in. **As of now `src/` contains only `README.md` placeholders — there is no implementation code.** Do not invent or assume an existing architecture; there is none yet.

## Intended commands (from README.md template)

These are the conventions a filled-in project is expected to follow — they do **not** run yet (no code):

```bash
conda env create -f environment.yml   # creates env "dl-project" (python 3.10, pytorch)
conda activate dl-project

python -m src.training.train    --config experiments/configs/baseline.yaml
python -m src.training.train    --config experiments/configs/model_v1.yaml
python -m src.evaluation.evaluate --config experiments/configs/model_v1.yaml
```

So when implementing code: training/eval are **module entrypoints** (`python -m src.training.train`) driven by **YAML configs** in `experiments/configs/`, not argparse-heavy CLIs. Mirror that pattern.

## Directory contract (from CONTRIBUTING.md)

- `src/datasets/` — download/parse scripts + `Dataset`/`DataLoader` definitions
- `src/models/` — `nn.Module` architectures only (no training logic)
- `src/training/` — loops, losses, optimizers
- `src/evaluation/` — inference + metrics
- `src/utils/` — loggers, viz, helpers
- `experiments/configs/` — YAML hyperparams/paths; `experiments/logs/` + `experiments/checkpoints/` are **gitignored**
- `data/` — gitignored; datasets never committed
- `notebooks/` — exploration/POC only; real training code goes in `src/`

Single-responsibility files: a model file must not contain the training loop, and vice versa.

## Conventions that matter here

- **Reproducibility**: keep `environment.yml` current when adding deps (conda packages in `dependencies:`, pip-only under the `pip:` block).
- **Type hints + docstrings** expected on main classes/functions; descriptive names (`compute_cross_entropy`, not `calc_ce`).
- **Git**: work on `feature/<name>` branches, not `main`, for multi-person work. Commit messages should be descriptive (`Add custom triplet loss in training loop`), not "update"/"fix".
- The report split: `README.md` = technical only (setup/run/results); `docs/REPORT.md` = theory, analysis, contribution breakdown, AI-usage declaration.
