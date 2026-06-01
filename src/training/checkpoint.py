"""Checkpoint saving / loading.

Model weights are stored in bf16 to keep checkpoints small. Two flavours:
``save_best_checkpoint`` (weights + metadata only) and ``save_last_checkpoint``
(also optimizer + scheduler state for resuming). :func:`load_pretrained_compressor`
loads only the compressor sub-module from an arbitrary checkpoint.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..models.reconstructor import QueryConditionedGTReconstructor


def unwrap_state_dict(obj: object) -> Dict[str, torch.Tensor]:
    """Return the raw ``state_dict`` from a checkpoint dict or bare state dict."""
    if isinstance(obj, dict):
        for key in ("model_state", "state_dict"):
            if key in obj:
                return obj[key]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj  # type: ignore[return-value]
    raise ValueError("Unrecognized checkpoint format.")


def _bf16_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    m = model.module if isinstance(model, nn.DataParallel) else model
    return {
        k: v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in m.state_dict().items()
    }


def save_best_checkpoint(
    path: str,
    model: nn.Module,
    epoch: int,
    global_step: int,
    best_val: float,
    config: dict,
) -> None:
    """Save the best-so-far model (weights + metadata)."""
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "model_state": _bf16_state(model),
            "config": config,
        },
        path,
    )


def save_last_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    epoch: int,
    global_step: int,
    best_val: float,
    config: dict,
) -> None:
    """Save the latest model plus optimizer / scheduler state for resuming."""
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_val": best_val,
            "model_state": _bf16_state(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "config": config,
        },
        path,
    )


def load_pretrained_compressor(
    model: QueryConditionedGTReconstructor, ckpt_path: str
) -> None:
    """Load only the compressor sub-module weights from ``ckpt_path``."""
    print(f"Loading pretrained compressor: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = unwrap_state_dict(ckpt)
    comp, direct = {}, {}
    for k, v in state.items():
        k = k.removeprefix("module.")
        (comp if k.startswith("compressor.") else direct)[
            k.removeprefix("compressor.")
        ] = v
    missing, unexpected = model.compressor.load_state_dict(
        comp if comp else direct, strict=False
    )
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")


def set_compressor_trainable(
    model: QueryConditionedGTReconstructor, trainable: bool
) -> None:
    """Freeze or unfreeze the compressor parameters."""
    for p in model.compressor.parameters():
        p.requires_grad = trainable
