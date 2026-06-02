"""Entrypoint: train query-conditioned GT visual-token reconstruction.

Architecture::

    context  -> TokenCompressor (xLSTM)  -> Z
    question -> frozen CLIP + proj        -> Q
    Q cross-attends Z -> Transformer Decoder -> pred_tokens [B, out_len, D]

Data is split at the *video* level (no leakage across train/val/test). The GT
window is sampled with :class:`GTWindowSampler` augmentation on train; val/test
use exact GT. All paths and hyperparameters live in a YAML config — there are no
hardcoded paths. Run::

    python -m src.training.train --config experiments/configs/train_local.yaml
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None

from ..datasets.annotations import read_annotations
from ..datasets.gt_window import GTWindowSampler
from ..datasets.reconstruction_dataset import (
    EgoTempoGTReconstructionDataset,
    build_collate_fn,
)
from ..datasets.token_index import TokenChunkIndex
from ..models.reconstructor import QueryConditionedGTReconstructor
from ..models.text_encoder import build_clip_tokenizer
from ..utils.config import load_config, parse_config_arg
from ..utils.seeding import count_params, safe_makedirs, set_seed
from .checkpoint import (
    load_pretrained_compressor,
    save_best_checkpoint,
    save_last_checkpoint,
    set_compressor_trainable,
    unwrap_state_dict,
)
from .losses import ReconstructionLoss


@dataclass
class TrainConfig:
    """All training parameters (paths + data + model + optim + loss + logging)."""

    # paths
    tokens_dir: str
    annotations: str
    checkpoint_dir: str
    pretrained_compressor: str = ""
    recursive_tokens: bool = False

    # device / runtime
    device: str = "auto"  # "auto" -> cuda if available else cpu
    dtype: str = "bf16"  # "bf16" | "fp32" (bf16 only used on cuda)
    num_workers: int = 8
    seed: int = 42

    # token loading
    pool_to_fps: int = 1

    # GT window augmentation
    gt_max_left_pad: float = 30.0
    gt_max_right_pad: float = 30.0
    gt_window_zero_dur_pad: float = 5.0
    gt_exact_prob: float = 0.2

    # CLIP
    clip_model: str = "openai/clip-vit-large-patch14"
    max_question_len: int = 77

    # data
    context_strategy: str = "full_until_query"  # full_until_query | full_video | random_crop
    random_crop_extra_min: int = 16
    random_crop_extra_max: int = 256
    max_seq_len: int = 2048
    out_len: int = 32
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # decode mode (Phase 2): "fixed" (out_len, time-warped) | "length_aware"
    decode_mode: str = "fixed"
    max_out_len: int = 64  # cap for length_aware decoding

    # model
    embed_dim: int = 4096
    num_heads: int = 8
    num_queries: int = 16
    decoder_layers: int = 2
    dropout: float = 0.1
    memory_mode: str = "z"  # z | z_m | h | h_z
    xlstm_num_blocks: int = 3
    xlstm_slstm_at: Sequence[int] = field(default_factory=lambda: [1])
    freeze_compressor: bool = False
    # compressor bottleneck (Phase 2): Perceiver resampler + multi-scale latents
    use_perceiver: bool = False
    perceiver_depth: int = 2
    multiscale_strides: Sequence[int] = field(default_factory=lambda: [1])

    # optimization
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    # loss
    mse_weight: float = 1.0
    cosine_weight: float = 0.1
    infonce_weight: float = 0.0
    infonce_temperature: float = 0.07
    # soft-DTW (Phase 1): shift-tolerant reconstruction term
    softdtw_weight: float = 0.0
    softdtw_gamma: float = 0.1
    softdtw_dist: str = "cosine"  # cosine | l2
    softdtw_normalize: bool = True
    # temporal hard negatives (Phase 1): same-video shifted windows for InfoNCE
    num_temporal_negatives: int = 0
    temporal_neg_max_iou: float = 0.5
    # length head (Phase 2, length_aware decode): weight on SmoothL1(len) term
    length_weight: float = 0.0
    # epochs to keep length_weight=0 before enabling it (0 = always active)
    length_warmup_epochs: int = 0

    # logging
    use_wandb: bool = False
    wandb_project: str = "query-gt-token-reconstruction"
    wandb_mode: str = "offline"  # online | offline | disabled
    # "loss" (lower=better) | "cos_seq" (higher=better)
    checkpoint_metric: str = "loss"

    # control flow
    dry_run: bool = False
    resume: str = ""


def resolve_device(name: str) -> torch.device:
    """Resolve ``"auto"`` to cuda when available, else cpu."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _move(
    batch: Dict[str, object], device: torch.device, dtype: torch.dtype
) -> Dict[str, object]:
    return {
        "context_tokens": batch["context_tokens"].to(device=device, dtype=dtype, non_blocking=True),  # type: ignore
        "context_mask": batch["context_mask"].to(device=device, non_blocking=True),  # type: ignore
        "question_ids": batch["question_ids"].to(device=device, non_blocking=True),  # type: ignore
        "question_mask": batch["question_mask"].to(device=device, non_blocking=True),  # type: ignore
        "target_tokens": batch["target_tokens"].to(device=device, dtype=dtype, non_blocking=True),  # type: ignore
        "target_mask": batch["target_mask"].to(device=device, non_blocking=True),  # type: ignore
        "target_len": batch["target_len"].to(device=device, non_blocking=True),  # type: ignore
        "negatives": batch["negatives"].to(device=device, dtype=dtype, non_blocking=True),  # type: ignore
        "negatives_mask": batch["negatives_mask"].to(device=device, non_blocking=True),  # type: ignore
        "metadata": batch["metadata"],
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: ReconstructionLoss,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[object],
    device: torch.device,
    dtype: torch.dtype,
    cfg: TrainConfig,
    epoch: int,
    train: bool,
    global_step: int,
) -> Tuple[float, Dict[str, float], int]:
    """Run one train or eval epoch; return ``(avg_loss, avg_metrics, global_step)``."""
    model.train(train)
    tot_loss = 0.0
    tot_met: Dict[str, float] = {}
    n = 0
    if train:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for bi, rb in enumerate(
            tqdm(dataloader, desc=f"{'Train' if train else 'Val'} {epoch}")
        ):
            b = _move(rb, device, dtype)
            pred, aux = model(
                b["context_tokens"],
                b["context_mask"],
                b["question_ids"],
                b["question_mask"],
            )
            negs = b["negatives"] if b["negatives"].shape[1] > 0 else None
            neg_mask = b["negatives_mask"] if negs is not None else None
            loss, met = loss_fn(
                pred,
                b["target_tokens"],
                target_mask=b["target_mask"],
                target_lengths=b["target_len"],
                pred_len=aux.get("pred_len"),
                negs=negs,
                neg_mask=neg_mask,
            )
            dl = float(loss.detach().cpu())
            if train:
                (loss / cfg.gradient_accumulation_steps).backward()
                if (bi + 1) % cfg.gradient_accumulation_steps == 0 or (bi + 1) == len(
                    dataloader
                ):
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if wandb and cfg.use_wandb:
                        wandb.log(
                            {
                                "train/loss_step": dl,
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "train/mse": met["mse"],
                                "train/cos_token": met["cos_token"],
                                "train/cos_seq": met["cos_seq"],
                            },
                            step=global_step,
                        )
            tot_loss += dl
            for k, v in met.items():
                tot_met[k] = tot_met.get(k, 0.0) + v
            n += 1

    avg = tot_loss / max(1, n)
    avg_m = {k: v / max(1, n) for k, v in tot_met.items()}
    return avg, avg_m, global_step


def build_splits(
    cfg: TrainConfig,
) -> Tuple[
    EgoTempoGTReconstructionDataset,
    Optional[EgoTempoGTReconstructionDataset],
    Optional[EgoTempoGTReconstructionDataset],
    TokenChunkIndex,
]:
    """Index tokens, read annotations, and build video-level train/val/test datasets."""
    print("\nIndexing tokens...")
    ci = TokenChunkIndex(
        cfg.tokens_dir, recursive=cfg.recursive_tokens, pool_to_fps=cfg.pool_to_fps
    )
    print(f"  .pt={ci.num_files} | videos={len(ci.by_video)} | skipped={ci.num_skipped}")

    print("\nReading annotations...")
    anns = read_annotations(cfg.annotations)
    print(f"  Loaded: {len(anns)}")

    # video-level split (no leakage across splits)
    all_video_ids = sorted(set(a.video_id for a in anns))
    random.shuffle(all_video_ids)
    n_total = len(all_video_ids)
    n_test = max(1, int(n_total * cfg.test_ratio))
    n_val = max(1, int(n_total * cfg.val_ratio))
    n_train = n_total - n_val - n_test

    train_videos = set(all_video_ids[:n_train])
    val_videos = set(all_video_ids[n_train : n_train + n_val])
    test_videos = set(all_video_ids[n_train + n_val :])

    t_ann = [a for a in anns if a.video_id in train_videos]
    v_ann = [a for a in anns if a.video_id in val_videos]
    s_ann = [a for a in anns if a.video_id in test_videos]

    print(f"\nVideo-level split (seed={cfg.seed}):")
    print(f"  Train : {len(train_videos)} videos | {len(t_ann)} annotations")
    print(f"  Val   : {len(val_videos)} videos | {len(v_ann)} annotations")
    print(f"  Test  : {len(test_videos)} videos | {len(s_ann)} annotations")

    gt_sampler = GTWindowSampler(
        max_left_pad=cfg.gt_max_left_pad,
        max_right_pad=cfg.gt_max_right_pad,
        zero_dur_pad=cfg.gt_window_zero_dur_pad,
        exact_prob=cfg.gt_exact_prob,
    )
    ds_kw = dict(
        chunk_index=ci,
        out_len=cfg.out_len,
        max_seq_len=cfg.max_seq_len,
        random_crop_extra_min=cfg.random_crop_extra_min,
        random_crop_extra_max=cfg.random_crop_extra_max,
        decode_mode=cfg.decode_mode,
        max_out_len=cfg.max_out_len,
        num_temporal_negatives=cfg.num_temporal_negatives,
        temporal_neg_max_iou=cfg.temporal_neg_max_iou,
    )

    print("\nBuilding datasets...")
    tr_ds = EgoTempoGTReconstructionDataset(
        t_ann, context_strategy=cfg.context_strategy, gt_sampler=gt_sampler, **ds_kw
    )
    vl_ds = (
        EgoTempoGTReconstructionDataset(
            v_ann, context_strategy="full_until_query", gt_sampler=None, **ds_kw
        )
        if v_ann
        else None
    )
    ts_ds = (
        EgoTempoGTReconstructionDataset(
            s_ann, context_strategy="full_until_query", gt_sampler=None, **ds_kw
        )
        if s_ann
        else None
    )
    return tr_ds, vl_ds, ts_ds, ci


def build_model(cfg: TrainConfig) -> QueryConditionedGTReconstructor:
    """Construct the reconstructor from the config."""
    return QueryConditionedGTReconstructor(
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        decoder_layers=cfg.decoder_layers,
        num_queries=cfg.num_queries,
        out_len=cfg.out_len,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        num_blocks=cfg.xlstm_num_blocks,
        slstm_at=list(cfg.xlstm_slstm_at),
        clip_model_name=cfg.clip_model,
        memory_mode=cfg.memory_mode,
        decode_mode=cfg.decode_mode,
        max_out_len=cfg.max_out_len,
        use_perceiver=cfg.use_perceiver,
        perceiver_depth=cfg.perceiver_depth,
        multiscale_strides=list(cfg.multiscale_strides),
    )


def run(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    safe_makedirs(cfg.checkpoint_dir)

    device = resolve_device(cfg.device)
    dtype = (
        torch.bfloat16
        if (cfg.dtype == "bf16" and device.type == "cuda")
        else torch.float32
    )

    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tr_ds, vl_ds, ts_ds, _ = build_splits(cfg)
    tok = build_clip_tokenizer(cfg.clip_model)
    cfn = build_collate_fn(tok, cfg.max_question_len)

    lkw = dict(
        batch_size=cfg.batch_size,
        collate_fn=cfn,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    tr_ldr = DataLoader(tr_ds, shuffle=True, **lkw)
    vl_ldr = DataLoader(vl_ds, shuffle=False, **lkw) if vl_ds else None
    ts_ldr = DataLoader(ts_ds, shuffle=False, **lkw) if ts_ds else None

    if cfg.dry_run:
        b = next(iter(tr_ldr))
        print("\n=== DRY RUN ===")
        print("[DATA — batch 0]")
        print(f"  context_tokens : {tuple(b['context_tokens'].shape)}")
        print(f"  context_mask   : {tuple(b['context_mask'].shape)}  valid_tokens={int(b['context_mask'][0].sum())}")
        print(f"  target_tokens  : {tuple(b['target_tokens'].shape)}")
        print(f"  target_mask    : {tuple(b['target_mask'].shape)}  valid_tokens={int(b['target_mask'][0].sum())}")
        print(f"  target_len     : {b['target_len'].tolist()}")
        print(f"  negatives      : {tuple(b['negatives'].shape)}  (B, K, L, D) — K={b['negatives'].shape[1]} negatives")
        print(f"  negatives_mask : {tuple(b['negatives_mask'].shape)}")
        print(f"  window_names   : {[m['window_name'] for m in b['metadata']]}")
        print(f"  metadata[0]    : {b['metadata'][0]}")
        print("\n[MODEL]")
        _dry_loss_fn = ReconstructionLoss(
            mse_weight=cfg.mse_weight,
            cosine_weight=cfg.cosine_weight,
            softdtw_weight=cfg.softdtw_weight,
            infonce_weight=cfg.infonce_weight,
            length_weight=cfg.length_weight,
            infonce_temperature=cfg.infonce_temperature,
            softdtw_gamma=cfg.softdtw_gamma,
            softdtw_dist=cfg.softdtw_dist,
            softdtw_normalize=cfg.softdtw_normalize,
        )
        _dry_model = build_model(cfg).to(device=device, dtype=dtype)
        _dry_model.train()
        print(f"  loss weights: mse={cfg.mse_weight} cos={cfg.cosine_weight} "
              f"dtw={cfg.softdtw_weight} infonce={cfg.infonce_weight} "
              f"length={cfg.length_weight}")
        print(f"  memory_mode={cfg.memory_mode}  decode_mode={cfg.decode_mode}  "
              f"use_perceiver={cfg.use_perceiver}")
        print("\n[FORWARD — 2 batches]")
        for _i, _rb in enumerate(tr_ldr):
            if _i >= 2:
                break
            _b = _move(_rb, device, dtype)
            _pred, _aux = _dry_model(
                _b["context_tokens"], _b["context_mask"],
                _b["question_ids"], _b["question_mask"],
            )
            _negs = _b["negatives"] if _b["negatives"].shape[1] > 0 else None
            _neg_mask = _b["negatives_mask"] if _negs is not None else None
            _loss, _info = _dry_loss_fn(
                _pred, _b["target_tokens"],
                target_mask=_b["target_mask"],
                target_lengths=_b["target_len"],
                pred_len=_aux.get("pred_len"),
                negs=_negs,
                neg_mask=_neg_mask,
            )
            _pred_len = _aux.get("pred_len")
            print(f"  batch {_i}:")
            print(f"    pred shape     : {tuple(_pred.shape)}")
            print(f"    pred_len       : {[round(float(x.detach()), 2) for x in _pred_len] if _pred_len is not None else 'N/A'}  (true: {_b['target_len'].tolist()})")
            print(f"    loss={float(_loss.detach()):.4f}  mse={_info['mse']:.4f}  "
                  f"cos_token={_info['cos_token']:.4f}  cos_seq={_info['cos_seq']:.4f}")
            print(f"    soft_dtw={_info['soft_dtw']:.4f}  info_nce={_info['info_nce']:.4f}  "
                  f"length={_info['length']:.4f}")
        print("\nDRY RUN OK")
        return

    loss_fn = ReconstructionLoss(
        mse_weight=cfg.mse_weight,
        cosine_weight=cfg.cosine_weight,
        softdtw_weight=cfg.softdtw_weight,
        infonce_weight=cfg.infonce_weight,
        length_weight=cfg.length_weight,
        infonce_temperature=cfg.infonce_temperature,
        softdtw_gamma=cfg.softdtw_gamma,
        softdtw_dist=cfg.softdtw_dist,
        softdtw_normalize=cfg.softdtw_normalize,
    )

    model = build_model(cfg)
    if cfg.pretrained_compressor:
        load_pretrained_compressor(model, cfg.pretrained_compressor)
    if cfg.freeze_compressor:
        print("Freezing compressor.")
        set_compressor_trainable(model, False)
    model = model.to(device=device, dtype=dtype)
    print(f"\n{count_params(model)}")
    if torch.cuda.device_count() > 1:
        print(f"DataParallel on {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    # optimizer with decoupled weight decay (no decay on biases / norms / embeddings)
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        target = (
            nodecay
            if any(x in name.lower() for x in ("bias", "norm", "ln", "embedding"))
            else decay
        )
        target.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
    )
    spe = math.ceil(len(tr_ldr) / cfg.gradient_accumulation_steps)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.learning_rate,
        total_steps=max(1, spe * cfg.epochs),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    if cfg.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb not installed.")
        os.environ["WANDB_MODE"] = cfg.wandb_mode
        os.environ.setdefault("WANDB_DIR", os.path.join(cfg.checkpoint_dir, "wandb_logs"))
        wandb.init(project=cfg.wandb_project, config=asdict(cfg))

    best_val = float("inf") if cfg.checkpoint_metric == "loss" else float("-inf")
    gs = 0
    start_epoch = 1
    last_path = os.path.join(cfg.checkpoint_dir, "query_gt_reconstructor_last.pth")
    best_path = os.path.join(cfg.checkpoint_dir, "query_gt_reconstructor_best.pth")

    if cfg.resume and os.path.exists(cfg.resume):
        print(f"\nResuming from: {cfg.resume}")
        try:
            ckpt = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        except Exception:
            ckpt = torch.load(cfg.resume, map_location="cpu")
        m = model.module if isinstance(model, nn.DataParallel) else model
        m.load_state_dict(unwrap_state_dict(ckpt))
        if "optimizer_state" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state") is not None:
            sch.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        gs = ckpt.get("global_step", 0)
        print(f"  Resumed epoch {start_epoch - 1} | best_val={best_val:.4f} | gs={gs}")
    elif cfg.resume:
        print(f"[WARN] --resume path not found: {cfg.resume}. Starting from scratch.")

    print(
        f"\nTraining: epochs {start_epoch}->{cfg.epochs} | {len(tr_ds)} examples "
        f"(max_left={cfg.gt_max_left_pad}s max_right={cfg.gt_max_right_pad}s "
        f"exact_prob={cfg.gt_exact_prob})"
    )

    config_dict = asdict(cfg)
    for epoch in range(start_epoch, cfg.epochs + 1):
        # length warmup: keep length_weight=0 until warmup epochs are done
        if cfg.length_warmup_epochs > 0:
            active_lw = 0.0 if epoch <= cfg.length_warmup_epochs else cfg.length_weight
            if loss_fn.length_weight != active_lw:
                loss_fn.length_weight = active_lw
                best_val = float("inf") if cfg.checkpoint_metric == "loss" else float("-inf")
                print(f"  [warmup] length_weight -> {active_lw} (epoch {epoch}) | best_val reset")
        tl, tm, gs = run_epoch(model, tr_ldr, loss_fn, opt, sch, device, dtype, cfg, epoch, True, gs)
        print(
            f"\nEpoch {epoch} train | loss={tl:.4f} mse={tm['mse']:.4f} "
            f"cos_token={tm['cos_token']:.4f} cos_seq={tm['cos_seq']:.4f}"
        )

        vl, vm = tl, tm
        if vl_ldr:
            vl, vm, _ = run_epoch(model, vl_ldr, loss_fn, None, None, device, dtype, cfg, epoch, False, gs)
            print(
                f"Epoch {epoch} val   | loss={vl:.4f} mse={vm['mse']:.4f} "
                f"cos_token={vm['cos_token']:.4f} cos_seq={vm['cos_seq']:.4f}"
            )

        if wandb and cfg.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": tl,
                    "train/mse": tm["mse"],
                    "train/cos_seq": tm["cos_seq"],
                    "val/loss": vl,
                    "val/mse": vm["mse"],
                    "val/cos_seq": vm["cos_seq"],
                },
                step=gs,
            )

        ckpt_score = vl if cfg.checkpoint_metric == "loss" else vm["cos_seq"]
        is_best = ckpt_score < best_val if cfg.checkpoint_metric == "loss" else ckpt_score > best_val
        if is_best:
            best_val = ckpt_score
            save_best_checkpoint(best_path, model, epoch, gs, best_val, config_dict)
            print(f"  Best saved: {best_path}  ({cfg.checkpoint_metric}={best_val:.4f})")

        save_last_checkpoint(last_path, model, opt, sch, epoch, gs, best_val, config_dict)
        print(f"  Last saved: {last_path}")

    # final test evaluation on the best checkpoint
    if ts_ldr:
        if os.path.exists(best_path):
            print(f"\nLoading best checkpoint for test: {best_path}")
            try:
                ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            except Exception:
                ckpt = torch.load(best_path, map_location="cpu")
            m = model.module if isinstance(model, nn.DataParallel) else model
            state = {
                k: v.to(dtype) if v.is_floating_point() else v
                for k, v in unwrap_state_dict(ckpt).items()
            }
            m.load_state_dict(state)
        sl, sm, _ = run_epoch(model, ts_ldr, loss_fn, None, None, device, dtype, cfg, cfg.epochs, False, gs)
        print(
            f"\nTEST | loss={sl:.4f} mse={sm['mse']:.4f} "
            f"cos_token={sm['cos_token']:.4f} cos_seq={sm['cos_seq']:.4f}"
        )
        if wandb and cfg.use_wandb:
            wandb.log({"test/loss": sl, "test/mse": sm["mse"], "test/cos_seq": sm["cos_seq"]}, step=gs)

    if wandb and cfg.use_wandb:
        wandb.finish()
    print(f"\nDone. Best val loss: {best_val:.4f}")


def main() -> None:
    config_path = parse_config_arg("Train query-conditioned GT token reconstruction.")
    cfg = load_config(TrainConfig, config_path)
    run(cfg)


if __name__ == "__main__":
    main()
