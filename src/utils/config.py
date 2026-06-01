"""YAML configuration loading.

All runtime parameters (paths, hyperparameters, device settings) live in YAML
files under ``experiments/configs/``. Code contains no hardcoded paths: every
entrypoint takes ``--config <file>`` and builds a typed config dataclass from it.

Two helpers are provided:
    - ``load_yaml``: read a YAML file into a plain dict.
    - ``from_dict``: populate a dataclass from a dict, ignoring unknown keys and
      keeping dataclass defaults for missing keys.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(data)}.")
    return data


def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Build a dataclass instance from a dict.

    Unknown keys are ignored (with a warning); missing keys fall back to the
    dataclass default. This keeps configs forward/backward compatible.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass.")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        print(f"[config] ignoring unknown keys: {sorted(unknown)}")
    kwargs = {k: v for k, v in data.items() if k in known}
    return cls(**kwargs)  # type: ignore[call-arg]


def load_config(cls: Type[T], path: str) -> T:
    """Load a YAML file and build a typed config dataclass from it."""
    return from_dict(cls, load_yaml(path))


def parse_config_arg(description: str) -> str:
    """Minimal CLI: every entrypoint exposes only ``--config <path>``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file (see experiments/configs/).",
    )
    args = parser.parse_args()
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    return args.config
