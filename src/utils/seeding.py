"""Reproducibility and parameter-counting helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> str:
    """Return a human-readable total / trainable parameter summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total params: {total / 1e6:.1f}M | Trainable: {trainable / 1e6:.1f}M"


def safe_makedirs(path: str) -> None:
    """Create a directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)
