"""Append-only JSON registry of ladder results (one entry per model / baseline).

The registry is the single source of truth for two things:
    1. **Resume** — when a ladder run is interrupted (e.g. cluster time limit),
       relaunching skips every entry already present and continues from the
       first missing step. No model weights are needed for this.
    2. **Reporting** — :mod:`src.evaluation.plot_progress` reads the registry to
       draw the cross-model "improvement ladder" chart.

Each entry is a dict::

    {
        "name": "02_mse_only",
        "kind": "model" | "baseline",
        "split": "test" | "val",
        "metrics": {"cos_seq": ..., "cos_token": ..., "mse": ..., "norm_ratio": ...},
        "epochs_run": 12,          # models only
        "stopped_early": true,     # models only
    }

Writes are atomic (write to a temp file, then ``os.replace``) so an interrupted
write can never corrupt the registry.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Optional

SCHEMA_VERSION = 1


def load_registry(path: str) -> Dict[str, object]:
    """Load the registry, returning an empty one if the file does not exist."""
    if not os.path.exists(path):
        return {"schema": SCHEMA_VERSION, "entries": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "entries" not in data:
        data["entries"] = []
    return data


def _atomic_write(path: str, data: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def has_entry(path: str, name: str) -> bool:
    """True if an entry with this ``name`` already exists (i.e. already done)."""
    reg = load_registry(path)
    return any(e.get("name") == name for e in reg["entries"])  # type: ignore[union-attr]


def completed_names(path: str) -> List[str]:
    """Return the list of names already recorded (preserves insertion order)."""
    reg = load_registry(path)
    return [e.get("name", "") for e in reg["entries"]]  # type: ignore[union-attr]


def append_entry(path: str, entry: Dict[str, object]) -> None:
    """Append (or replace by name) one entry and persist atomically."""
    reg = load_registry(path)
    entries: List[Dict[str, object]] = reg["entries"]  # type: ignore[assignment]
    entries = [e for e in entries if e.get("name") != entry.get("name")]
    entries.append(entry)
    reg["entries"] = entries
    reg["schema"] = SCHEMA_VERSION
    _atomic_write(path, reg)


def make_entry(
    name: str,
    kind: str,
    split: str,
    metrics: Dict[str, float],
    epochs_run: Optional[int] = None,
    stopped_early: Optional[bool] = None,
) -> Dict[str, object]:
    """Build a registry entry dict with a consistent shape."""
    entry: Dict[str, object] = {
        "name": name,
        "kind": kind,
        "split": split,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    if epochs_run is not None:
        entry["epochs_run"] = int(epochs_run)
    if stopped_early is not None:
        entry["stopped_early"] = bool(stopped_early)
    return entry
