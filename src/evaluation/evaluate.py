"""Entrypoint: evaluate a trained reconstructor on the held-out test split.

Reuses the same video-level split and model construction as training, loads a
checkpoint, and reports reconstruction metrics (MSE, per-token cosine, sequence
cosine) on the test set. Uses the identical YAML config as training.

Run::

    python -m src.evaluation.evaluate --config experiments/configs/train_local.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets.reconstruction_dataset import build_collate_fn
from ..models.text_encoder import build_clip_tokenizer
from ..training.checkpoint import unwrap_state_dict
from ..training.train import TrainConfig, _move, build_model, build_splits, resolve_device
from ..utils.config import load_config, parse_config_arg
from ..utils.seeding import set_seed


@dataclass
class EvalConfig(TrainConfig):
    """Training config plus the checkpoint to evaluate."""

    # null/empty -> <checkpoint_dir>/query_gt_reconstructor_best.pth
    eval_checkpoint: str = ""
    eval_split: str = "test"  # test | val


@torch.no_grad()
def evaluate(model, loader, device, dtype) -> Dict[str, float]:
    """Compute mean MSE / per-token cosine / sequence cosine over the loader."""
    model.eval()
    agg = {"mse": 0.0, "cos_token": 0.0, "cos_seq": 0.0}
    n = 0
    for rb in tqdm(loader, desc="eval", leave=False):
        b = _move(rb, device, dtype)
        pred, _ = model(
            b["context_tokens"],
            b["context_mask"],
            b["question_ids"],
            b["question_mask"],
        )
        pred = pred.float()
        target = b["target_tokens"].float()
        if pred.shape[1] > target.shape[1]:  # length_aware: align time axis
            pred = pred[:, : target.shape[1]]
        mask = b["target_mask"]  # [B, L] — score only real tokens
        m = mask.unsqueeze(-1).float()
        agg["mse"] += float(
            ((((pred - target) ** 2) * m).sum() / m.sum().clamp_min(1e-6) / pred.shape[-1]).cpu()
        )
        cs = F.cosine_similarity(pred, target, dim=-1)  # [B, L]
        agg["cos_token"] += float(
            ((cs * mask.float()).sum() / mask.float().sum().clamp_min(1e-6)).cpu()
        )
        pm = mask.unsqueeze(-1).to(pred.dtype)
        p_mean = (pred * pm).sum(1) / pm.sum(1).clamp_min(1e-6)
        t_mean = (target * pm).sum(1) / pm.sum(1).clamp_min(1e-6)
        agg["cos_seq"] += float(
            F.cosine_similarity(p_mean, t_mean, dim=-1).mean().cpu()
        )
        n += 1
    return {k: v / max(1, n) for k, v in agg.items()}


def run(cfg: EvalConfig) -> None:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = (
        torch.bfloat16
        if (cfg.dtype == "bf16" and device.type == "cuda")
        else torch.float32
    )
    print(f"Device: {device}")

    tr_ds, vl_ds, ts_ds, _ = build_splits(cfg)
    ds = ts_ds if cfg.eval_split == "test" else vl_ds
    if ds is None:
        raise ValueError(f"No data for eval_split={cfg.eval_split!r}.")

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

    model = build_model(cfg).to(device=device, dtype=dtype)

    ckpt_path = cfg.eval_checkpoint or (
        f"{cfg.checkpoint_dir.rstrip('/')}/query_gt_reconstructor_best.pth"
    )
    print(f"Loading checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {
        k: v.to(dtype) if v.is_floating_point() else v
        for k, v in unwrap_state_dict(ckpt).items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")

    metrics = evaluate(model, loader, device, dtype)
    print("=" * 40)
    print(f"split: {cfg.eval_split}")
    for k, v in metrics.items():
        print(f"{k:>12}: {v:.4f}")


def main() -> None:
    config_path = parse_config_arg("Evaluate a trained GT token reconstructor.")
    cfg = load_config(EvalConfig, config_path)
    run(cfg)


if __name__ == "__main__":
    main()
