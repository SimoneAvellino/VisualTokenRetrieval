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
from .losses import MeanInfoNCELoss, ReconstructionLoss


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
    # "query_gt"  -> QueryConditionedGTReconstructor (full sequence reconstruction)
    # "mean"      -> MeanReconstructor (single pooled-vector prediction, simplified)
    model_type: str = "query_gt"
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
    # internal working dimension (0 = use embed_dim, i.e. full-size model). A
    # smaller value (e.g. 1024) adds in/out projections and shrinks the model.
    model_dim: int = 0

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
    norm_weight: float = 0.0      # per-token norm matching (fixes scale collapse)
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
    # span head: weight on the temporal-grounding loss (L1 + 1D IoU on the moment's
    # start/end fractions). The direct localization signal.
    span_weight: float = 0.0
    # mean model (model_type="mean"): in-batch InfoNCE on pooled vectors + cosine
    # anchor. InfoNCE needs batch_size > 1 to have negatives.
    batch_infonce_weight: float = 1.0
    batch_infonce_temperature: float = 0.07
    batch_infonce_symmetric: bool = True

    # checkpoint selection: "loss" (lower=better) | "cos_seq" (higher=better)
    checkpoint_metric: str = "loss"

    # early stopping: stop after this many epochs without val-metric improvement
    # (0 disables). The best checkpoint is always kept, so nothing is lost.
    early_stopping_patience: int = 0

    # ladder integration
    run_name: str = ""              # label used in logs / result registry
    keep_checkpoints: bool = True   # ladder sets False to delete weights after test
    loss_curve_path: str = ""       # if set, write the train/val loss curve here
    pca_scatter_path: str = ""      # if set, write the pred-vs-GT PCA scatter here
    max_train_batches: int = 0      # >0 caps train batches per epoch (dry-run)
    max_eval_batches: int = 0       # >0 caps eval batches per epoch (dry-run)
    ablation_shuffle_query: bool = False  # also run retrieval with queries shuffled
                                          # across samples — if recall barely drops,
                                          # the model ignores the query.

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
        "target_span": batch["target_span"].to(device=device, non_blocking=True),  # type: ignore  (keep float32)
        "negatives": batch["negatives"].to(device=device, dtype=dtype, non_blocking=True),  # type: ignore
        "negatives_mask": batch["negatives_mask"].to(device=device, non_blocking=True),  # type: ignore
        "metadata": batch["metadata"],
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[object],
    device: torch.device,
    dtype: torch.dtype,
    cfg: TrainConfig,
    epoch: int,
    train: bool,
    global_step: int,
    step_log: Optional[list] = None,
) -> Tuple[float, Dict[str, float], int]:
    """Run one train or eval epoch; return ``(avg_loss, avg_metrics, global_step)``.

    If ``step_log`` is provided (train only), each optimizer step appends
    ``(global_step, loss)`` to it for the per-step loss curve.
    """
    model.train(train)
    tot_loss = 0.0
    tot_met: Dict[str, float] = {}
    n = 0
    max_batches = cfg.max_train_batches if train else cfg.max_eval_batches
    if train:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for bi, rb in enumerate(
            tqdm(dataloader, desc=f"{'Train' if train else 'Val'} {epoch}")
        ):
            if max_batches and bi >= max_batches:
                break
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
                target_span=b.get("target_span"),
                pred_span=aux.get("pred_span"),
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
                    if step_log is not None:
                        step_log.append((global_step, dl))
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

    # video-level split (no leakage across splits). Use a dedicated RNG seeded by
    # cfg.seed so the split is identical for baselines and every model regardless
    # of global RNG state or call order — otherwise the ladder would compare them
    # on different test sets.
    all_video_ids = sorted(set(a.video_id for a in anns))
    random.Random(cfg.seed).shuffle(all_video_ids)
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


def build_loss(cfg: TrainConfig) -> nn.Module:
    """Construct the loss from the config (model_type selects which)."""
    if cfg.model_type == "mean":
        return MeanInfoNCELoss(
            infonce_weight=cfg.batch_infonce_weight,
            cosine_weight=cfg.cosine_weight,
            temperature=cfg.batch_infonce_temperature,
            symmetric=cfg.batch_infonce_symmetric,
        )
    return ReconstructionLoss(
        mse_weight=cfg.mse_weight,
        cosine_weight=cfg.cosine_weight,
        norm_weight=cfg.norm_weight,
        softdtw_weight=cfg.softdtw_weight,
        infonce_weight=cfg.infonce_weight,
        length_weight=cfg.length_weight,
        span_weight=cfg.span_weight,
        infonce_temperature=cfg.infonce_temperature,
        softdtw_gamma=cfg.softdtw_gamma,
        softdtw_dist=cfg.softdtw_dist,
        softdtw_normalize=cfg.softdtw_normalize,
    )


def build_model(cfg: TrainConfig) -> nn.Module:
    """Construct the reconstructor from the config (model_type selects which)."""
    if cfg.model_type == "mean":
        from ..models.mean_reconstructor import MeanReconstructor

        return MeanReconstructor(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            decoder_layers=cfg.decoder_layers,
            dropout=cfg.dropout,
            num_blocks=cfg.xlstm_num_blocks,
            slstm_at=list(cfg.xlstm_slstm_at),
            max_seq_len=cfg.max_seq_len,
            num_queries=cfg.num_queries,
            clip_model_name=cfg.clip_model,
        )
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
        model_dim=cfg.model_dim,
    )


# metrics usable for checkpoint selection / early stopping (all higher=better).
# cross-sample (cross-video) retrieval:
_RETRIEVAL_CKPT_METRICS = ("recall@1", "recall@5", "mrr")
# intra-video localization (the project goal: find the moment WITHIN the video):
_LOCAL_CKPT_METRICS = ("intra_recall@1", "intra_mrr")


def _model_predict_fn(model, device, dtype):
    """A predict_fn (batch -> pred) bound to the unwrapped, eval-mode model."""
    m = model.module if isinstance(model, nn.DataParallel) else model
    m.eval()
    return m, (lambda b: m(
        b["context_tokens"], b["context_mask"],
        b["question_ids"], b["question_mask"],
    )[0])


@torch.no_grad()
def _val_retrieval(model, loader, device, dtype, cfg) -> Dict[str, float]:
    """Cross-sample (cross-video) retrieval scores on a loader."""
    from ..evaluation.retrieval import collect_vectors, retrieval_scores

    _, pred_fn = _model_predict_fn(model, device, dtype)
    pv, gv, _ = collect_vectors(
        predict_fn=pred_fn,
        move_fn=lambda rb: _move(rb, device, dtype),
        loader=loader,
        max_batches=cfg.max_eval_batches,
    )
    return retrieval_scores(pv, gv)


@torch.no_grad()
def _val_localization(model, loader, device, dtype, cfg) -> Dict[str, float]:
    """Intra-video localization scores (rank the GT moment among same-video
    temporal negatives) — aligned with the goal of locating the moment in the
    given video. Needs the loader to provide temporal negatives."""
    from ..evaluation.retrieval import intra_video_scores

    _, pred_fn = _model_predict_fn(model, device, dtype)
    return intra_video_scores(
        predict_fn=pred_fn,
        move_fn=lambda rb: _move(rb, device, dtype),
        loader=loader,
        max_batches=cfg.max_eval_batches,
    )


def run(cfg: TrainConfig) -> Optional[Dict[str, object]]:
    """Train one model and return its result dict (None on dry-run)."""
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
        _dry_loss_fn = build_loss(cfg)
        _dry_model = build_model(cfg).to(device=device, dtype=dtype)
        _dry_model.train()
        print(f"  loss weights: mse={cfg.mse_weight} cos={cfg.cosine_weight} "
              f"norm={cfg.norm_weight} dtw={cfg.softdtw_weight} "
              f"infonce={cfg.infonce_weight} length={cfg.length_weight}")
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
                target_span=_b.get("target_span"),
                pred_span=_aux.get("pred_span"),
            )
            _pred_len = _aux.get("pred_len")
            print(f"  batch {_i}:")
            print(f"    pred shape     : {tuple(_pred.shape)}")
            print(f"    pred_len       : {[round(float(x.detach()), 2) for x in _pred_len] if _pred_len is not None else 'N/A'}  (true: {_b['target_len'].tolist()})")
            print(f"    loss={float(_loss.detach()):.4f}  mse={_info['mse']:.4f}  "
                  f"cos_token={_info['cos_token']:.4f}  cos_seq={_info['cos_seq']:.4f}  "
                  f"norm={_info['norm']:.4f}")
            print(f"    soft_dtw={_info['soft_dtw']:.4f}  info_nce={_info['info_nce']:.4f}  "
                  f"length={_info['length']:.4f}")
        print("\nDRY RUN OK")
        return

    loss_fn = build_loss(cfg)

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
    run_name = cfg.run_name or "model"
    # in-memory loss history (per-step train, per-epoch val) for the loss curve.
    history: Dict[str, list] = {"train_steps": [], "val_epochs": []}
    epochs_since_improve = 0
    stopped_early = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        last_epoch = epoch
        # length warmup: keep length_weight=0 until warmup epochs are done
        if cfg.length_warmup_epochs > 0:
            active_lw = 0.0 if epoch <= cfg.length_warmup_epochs else cfg.length_weight
            if loss_fn.length_weight != active_lw:
                loss_fn.length_weight = active_lw
                best_val = float("inf") if cfg.checkpoint_metric == "loss" else float("-inf")
                epochs_since_improve = 0  # loss surface changed; restart patience
                print(f"  [warmup] length_weight -> {active_lw} (epoch {epoch}) | best_val reset")
        tl, tm, gs = run_epoch(
            model, tr_ldr, loss_fn, opt, sch, device, dtype, cfg, epoch, True, gs,
            step_log=history["train_steps"],
        )
        print(
            f"\nEpoch {epoch} train | loss={tl:.4f} mse={tm['mse']:.4f} "
            f"cos_token={tm['cos_token']:.4f} cos_seq={tm['cos_seq']:.4f}\n"
            f"             terms | norm={tm['norm']:.4f} soft_dtw={tm['soft_dtw']:.4f} "
            f"info_nce={tm['info_nce']:.4f} length={tm['length']:.4f} "
            f"span_iou={tm.get('span_iou', 0.0):.4f} norm_ratio={tm['norm_ratio']:.4f}"
        )

        # under a hard GPU quota (e.g. SLURM --gres=shard), the train-time cached
        # blocks otherwise stay reserved and the (lighter) val pass tips the
        # process over the cap. Report the true peak and release the cache first.
        if device.type == "cuda":
            print(f"             vram  | peak={torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
            torch.cuda.empty_cache()

        vl, vm = tl, tm
        if vl_ldr:
            vl, vm, _ = run_epoch(model, vl_ldr, loss_fn, None, None, device, dtype, cfg, epoch, False, gs)
            print(
                f"Epoch {epoch} val   | loss={vl:.4f} mse={vm['mse']:.4f} "
                f"cos_token={vm['cos_token']:.4f} cos_seq={vm['cos_seq']:.4f}\n"
                f"             terms | norm={vm['norm']:.4f} soft_dtw={vm['soft_dtw']:.4f} "
                f"info_nce={vm['info_nce']:.4f} length={vm['length']:.4f} "
                f"span_iou={vm.get('span_iou', 0.0):.4f} norm_ratio={vm['norm_ratio']:.4f}"
            )
        history["val_epochs"].append((gs, vl))

        # checkpoint / early-stopping score. "loss" (lower=better) and "cos_seq"
        # (higher) read the val pass; the retrieval metrics need an extra cross-
        # sample pass — they select against mode collapse, which the loss does not.
        if cfg.checkpoint_metric in _LOCAL_CKPT_METRICS and vl_ldr:
            vloc = _val_localization(model, vl_ldr, device, dtype, cfg)
            ckpt_score = vloc[cfg.checkpoint_metric]
            print(
                f"             val localization | intra_recall@1={vloc['intra_recall@1']:.4f} "
                f"intra_mrr={vloc['intra_mrr']:.4f} median_rank={vloc['intra_median_rank']:.1f} "
                f"(select on {cfg.checkpoint_metric})"
            )
        elif cfg.checkpoint_metric in _RETRIEVAL_CKPT_METRICS and vl_ldr:
            vretr = _val_retrieval(model, vl_ldr, device, dtype, cfg)
            ckpt_score = vretr[cfg.checkpoint_metric]
            print(
                f"             val retrieval | recall@1={vretr['recall@1']:.4f} "
                f"recall@5={vretr['recall@5']:.4f} mrr={vretr['mrr']:.4f} "
                f"(select on {cfg.checkpoint_metric})"
            )
        elif cfg.checkpoint_metric == "span_iou":
            ckpt_score = vm.get("span_iou", 0.0)   # from the val pass (span head)
        elif cfg.checkpoint_metric == "cos_seq":
            ckpt_score = vm["cos_seq"]
        else:
            ckpt_score = vl
        lower_better = cfg.checkpoint_metric == "loss"
        is_best = ckpt_score < best_val if lower_better else ckpt_score > best_val
        if is_best:
            best_val = ckpt_score
            epochs_since_improve = 0
            save_best_checkpoint(best_path, model, epoch, gs, best_val, config_dict)
            print(f"  Best saved: {best_path}  ({cfg.checkpoint_metric}={best_val:.4f})")
        else:
            epochs_since_improve += 1

        save_last_checkpoint(last_path, model, opt, sch, epoch, gs, best_val, config_dict)
        print(f"  Last saved: {last_path}")

        # early stopping: overfitting shows up as val metric no longer improving.
        if cfg.early_stopping_patience > 0 and epochs_since_improve >= cfg.early_stopping_patience:
            stopped_early = True
            print(
                f"\nEarly stopping at epoch {epoch}: no improvement for "
                f"{epochs_since_improve} epochs (patience={cfg.early_stopping_patience})."
            )
            break

    # final evaluation on the best checkpoint (test split if available, else val).
    eval_ldr = ts_ldr or vl_ldr
    eval_split = "test" if ts_ldr else "val"
    final_metrics: Dict[str, float] = {}
    if eval_ldr:
        if os.path.exists(best_path):
            print(f"\nLoading best checkpoint for {eval_split}: {best_path}")
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
        sl, sm, _ = run_epoch(model, eval_ldr, loss_fn, None, None, device, dtype, cfg, last_epoch, False, gs)
        final_metrics = {
            "loss": sl,
            "mse": sm["mse"],
            "cos_token": sm["cos_token"],
            "cos_seq": sm["cos_seq"],
            "norm_ratio": sm["norm_ratio"],
        }
        print(
            f"\n{eval_split.upper()} | loss={sl:.4f} mse={sm['mse']:.4f} "
            f"cos_token={sm['cos_token']:.4f} cos_seq={sm['cos_seq']:.4f} "
            f"norm_ratio={sm['norm_ratio']:.4f}"
        )

        # retrieval metrics + PCA scatter (cross-sample): does the model rank its
        # own GT above other moments? cos_token saturates, this does not.
        from ..evaluation.retrieval import collect_vectors, retrieval_scores

        m_eval = model.module if isinstance(model, nn.DataParallel) else model
        m_eval.eval()
        pred_vecs, gt_vecs, _ = collect_vectors(
            predict_fn=lambda b: m_eval(
                b["context_tokens"], b["context_mask"],
                b["question_ids"], b["question_mask"],
            )[0],
            move_fn=lambda rb: _move(rb, device, dtype),
            loader=eval_ldr,
            max_batches=cfg.max_eval_batches,
        )
        retr = retrieval_scores(pred_vecs, gt_vecs)
        final_metrics.update({
            "recall@1": retr["recall@1"],
            "recall@5": retr["recall@5"],
            "mrr": retr["mrr"],
            "median_rank": retr["median_rank"],
            "chance_r1": retr["chance_r1"],
        })
        print(
            f"{eval_split.upper()} retrieval | recall@1={retr['recall@1']:.4f} "
            f"recall@5={retr['recall@5']:.4f} mrr={retr['mrr']:.4f} "
            f"median_rank={retr['median_rank']:.1f} (chance r@1={retr['chance_r1']:.4f})"
        )

        # intra-video localization (the project goal): rank the GT moment among the
        # SAME video's other moments (temporal negatives).
        from ..evaluation.retrieval import intra_video_scores

        loc = intra_video_scores(
            predict_fn=lambda b: m_eval(
                b["context_tokens"], b["context_mask"],
                b["question_ids"], b["question_mask"],
            )[0],
            move_fn=lambda rb: _move(rb, device, dtype),
            loader=eval_ldr,
            max_batches=cfg.max_eval_batches,
        )
        final_metrics.update({
            "intra_recall@1": loc["intra_recall@1"],
            "intra_mrr": loc["intra_mrr"],
            "intra_median_rank": loc["intra_median_rank"],
        })
        print(
            f"{eval_split.upper()} localization | intra_recall@1={loc['intra_recall@1']:.4f} "
            f"intra_mrr={loc['intra_mrr']:.4f} median_rank={loc['intra_median_rank']:.1f} "
            f"(over same-video moments)"
        )

        # temporal-grounding span: predicted (start, end) vs GT, IoU-based —
        # the direct localization output ("return the moment" as a time span).
        span_ious = []
        for rb in eval_ldr:
            b = _move(rb, device, dtype)
            _, aux_s = m_eval(
                b["context_tokens"], b["context_mask"],
                b["question_ids"], b["question_mask"],
            )
            ps = aux_s.get("pred_span")
            if ps is None:
                break
            ps = ps.float()
            ts = b["target_span"].float()
            inter = (torch.min(ps[:, 1], ts[:, 1]) - torch.max(ps[:, 0], ts[:, 0])).clamp_min(0.0)
            union = (ps[:, 1] - ps[:, 0]) + (ts[:, 1] - ts[:, 0]) - inter
            span_ious.append((inter / union.clamp_min(1e-6)).cpu())
        if span_ious:
            iou = torch.cat(span_ious)
            final_metrics.update({
                "span_miou": float(iou.mean()),
                "span_recall@0.3": float((iou >= 0.3).float().mean()),
                "span_recall@0.5": float((iou >= 0.5).float().mean()),
                "span_recall@0.7": float((iou >= 0.7).float().mean()),
            })
            print(
                f"{eval_split.upper()} span grounding | mIoU={float(iou.mean()):.4f} "
                f"R@0.3={float((iou >= 0.3).float().mean()):.4f} "
                f"R@0.5={float((iou >= 0.5).float().mean()):.4f} "
                f"R@0.7={float((iou >= 0.7).float().mean()):.4f}"
            )

        # query-usage ablation: re-run retrieval with each prediction conditioned
        # on another sample's question. A small drop vs the real-query scores means
        # the model is not using the query to localise the moment.
        if cfg.ablation_shuffle_query:
            from ..evaluation.retrieval import shuffle_batch_questions

            def _predict_shuffled(b):
                s = shuffle_batch_questions(b)
                return m_eval(
                    s["context_tokens"], s["context_mask"],
                    s["question_ids"], s["question_mask"],
                )[0]

            sh_pred, sh_gt, _ = collect_vectors(
                predict_fn=_predict_shuffled,
                move_fn=lambda rb: _move(rb, device, dtype),
                loader=eval_ldr,
                max_batches=cfg.max_eval_batches,
            )
            retr_sh = retrieval_scores(sh_pred, sh_gt)
            final_metrics.update({
                "recall@1_shuffled": retr_sh["recall@1"],
                "recall@5_shuffled": retr_sh["recall@5"],
                "mrr_shuffled": retr_sh["mrr"],
            })
            d1 = retr["recall@1"] - retr_sh["recall@1"]
            print(
                f"{eval_split.upper()} retrieval [shuffled query] | "
                f"recall@1={retr_sh['recall@1']:.4f} recall@5={retr_sh['recall@5']:.4f} "
                f"mrr={retr_sh['mrr']:.4f}  -> Δrecall@1={d1:+.4f} "
                f"({'query USED' if d1 > 0.02 else 'query largely IGNORED'})"
            )
        if cfg.pca_scatter_path:
            from ..evaluation.plot_space import plot_pred_vs_gt

            plot_pred_vs_gt(gt_vecs, pred_vecs, cfg.pca_scatter_path, run_name, seed=cfg.seed)

    # render the loss curve image (numbers are not persisted, only the PNG).
    if cfg.loss_curve_path:
        from ..evaluation.plot_curves import plot_loss_curve

        plot_loss_curve(history, cfg.loss_curve_path, run_name, stopped_early=stopped_early)

    # the ladder does not need weights — only scores. Drop them to save disk.
    if not cfg.keep_checkpoints:
        for p in (best_path, last_path):
            if os.path.exists(p):
                os.remove(p)
        print("Removed checkpoints (keep_checkpoints=False).")

    print(f"\nDone. Best val {cfg.checkpoint_metric}: {best_val:.4f}")
    return {
        "name": run_name,
        "kind": "model",
        "split": eval_split,
        "metrics": final_metrics,
        "epochs_run": last_epoch,
        "stopped_early": stopped_early,
    }


def main() -> None:
    config_path = parse_config_arg("Train query-conditioned GT token reconstruction.")
    cfg = load_config(TrainConfig, config_path)
    run(cfg)


if __name__ == "__main__":
    main()
