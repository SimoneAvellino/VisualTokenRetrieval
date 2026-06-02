"""Supervised dataset + collate for query-conditioned GT token reconstruction.

Each item yields a context token sequence, a question string, and the target GT
tokens. Two target regimes are supported:

    - ``decode_mode="fixed"`` (legacy): the GT window is resampled to exactly
      ``out_len`` tokens (time-warped).
    - ``decode_mode="length_aware"``: the GT keeps its native token count (only
      down-sampled if it exceeds ``max_out_len``), so the target stays on the real
      time grid. Collate pads variable-length targets and emits a mask + length.

When ``num_temporal_negatives > 0`` (train only), each item also carries ``K``
**temporal hard negatives**: windows from the *same video*, shifted off the GT
span. Same scene/objects, wrong moment — these force the contrastive loss to learn
the action boundary rather than merely the video identity.

Training uses :class:`GTWindowSampler` augmentation; validation passes
``gt_sampler=None`` for exact GT windows (and no negatives).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from transformers import CLIPTokenizer

from .annotations import MomentAnnotation
from .gt_window import GTWindowSampler
from .token_index import TokenChunkIndex, resample_tokens


def _temporal_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Intersection-over-union of two ``(start, end)`` time spans."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


class EgoTempoGTReconstructionDataset(Dataset):
    """GT visual-token reconstruction with random window augmentation.

    At each ``__getitem__`` the GT window is sampled from the space of valid
    windows containing ``[gt_start, gt_end]`` via :class:`GTWindowSampler`, so
    every epoch sees a different window per annotation. Passing
    ``gt_sampler=None`` disables augmentation (validation: always exact GT).
    """

    def __init__(
        self,
        annotations: Sequence[MomentAnnotation],
        chunk_index: TokenChunkIndex,
        out_len: int,
        max_seq_len: int,
        context_strategy: str,
        random_crop_extra_min: int,
        random_crop_extra_max: int,
        gt_sampler: Optional[GTWindowSampler] = None,
        skip_uncovered: bool = True,
        decode_mode: str = "fixed",
        max_out_len: int = 64,
        num_temporal_negatives: int = 0,
        temporal_neg_max_iou: float = 0.5,
    ) -> None:
        self.chunk_index = chunk_index
        self.out_len = out_len
        self.max_seq_len = max_seq_len
        self.context_strategy = context_strategy
        self.random_crop_extra_min = random_crop_extra_min
        self.random_crop_extra_max = random_crop_extra_max
        # None = no augmentation (val mode): always use exact GT
        self.gt_sampler = gt_sampler
        self.decode_mode = decode_mode
        self.max_out_len = max_out_len
        # negatives are a training-only signal (needs the augmentation sampler)
        self.num_negs = num_temporal_negatives if gt_sampler is not None else 0
        self.neg_max_iou = temporal_neg_max_iou

        valid: List[MomentAnnotation] = []
        skipped = 0
        for ann in annotations:
            if not chunk_index.has_video(ann.video_id):
                skipped += 1
                continue
            cs = ann.gt_start - 1.0
            ce = ann.gt_end + 1.0 if ann.gt_end > ann.gt_start else ann.gt_start + 1.0
            if skip_uncovered and not chunk_index.overlaps_range(ann.video_id, cs, ce):
                skipped += 1
                continue
            valid.append(ann)

        if not valid:
            raise ValueError("No annotations have GT tokens on disk.")

        self.annotations = valid
        mode = (
            f"random window aug (max_left={gt_sampler.max_left_pad}s "
            f"max_right={gt_sampler.max_right_pad}s "
            f"exact_prob={gt_sampler.exact_prob})"
            if gt_sampler is not None
            else "exact GT (no augmentation)"
        )
        print(
            f"Dataset: {len(valid)} annotations | {skipped} skipped | mode={mode} "
            f"| decode={decode_mode} | negs={self.num_negs}"
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def _resample_target(self, tok: torch.Tensor) -> torch.Tensor:
        """Convert loaded GT tokens to the target tensor for the active decode mode.

        ``fixed``        -> always ``out_len`` (time-warped).
        ``length_aware`` -> native length, down-sampled only if it exceeds
                            ``max_out_len`` (keeps the real time grid otherwise).
        """
        if self.decode_mode == "fixed":
            return resample_tokens(tok, self.out_len)
        if tok.shape[0] > self.max_out_len:
            return resample_tokens(tok, self.max_out_len)
        return tok.float()

    def _get_ann_window(self, idx: int) -> Tuple[MomentAnnotation, float, float, str]:
        ann = self.annotations[idx]
        if self.gt_sampler is None:
            # val mode: exact GT, handle zero-duration
            if ann.gt_end <= ann.gt_start:
                zdp = 5.0
                return ann, max(0.0, ann.gt_start - zdp), ann.gt_end + zdp, "gt_exact"
            return ann, ann.gt_start, ann.gt_end, "gt_exact"
        ws, we, name = self.gt_sampler.sample(
            ann.gt_start, ann.gt_end, video_end=ann.query_time
        )
        return ann, ws, we, name

    def _sample_temporal_negatives(
        self, ann: MomentAnnotation, ws: float, we: float
    ) -> List[torch.Tensor]:
        """Sample up to ``num_negs`` same-video windows shifted off the GT span.

        Shifts are multiples of the window duration in both directions. A
        candidate is rejected if it falls outside the video, has no tokens on
        disk, or overlaps the GT moment above ``neg_max_iou`` (too easy / too
        similar to the positive).
        """
        dur = max(1.0, we - ws)
        gt_span = (ann.gt_start, ann.gt_end if ann.gt_end > ann.gt_start else ann.gt_start + 1.0)
        shifts = [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0]
        random.shuffle(shifts)
        negs: List[torch.Tensor] = []
        for s in shifts:
            if len(negs) >= self.num_negs:
                break
            ns = ws + s * dur
            ne = ns + dur
            if ns < 0:
                continue
            if ann.query_time is not None and ne > ann.query_time:
                continue
            if _temporal_iou((ns, ne), gt_span) > self.neg_max_iou:
                continue
            tok, _ = self.chunk_index.load_range(ann.video_id, ns, ne)
            if tok.numel() == 0:
                continue
            negs.append(self._resample_target(tok))
        return negs

    def _context_range(
        self, ann: MomentAnnotation
    ) -> Tuple[Optional[float], Optional[float]]:
        if self.context_strategy == "full_video":
            return None, None
        if self.context_strategy == "full_until_query":
            return None, ann.query_time
        if self.context_strategy == "random_crop":
            l = random.randint(self.random_crop_extra_min, self.random_crop_extra_max)
            r = random.randint(self.random_crop_extra_min, self.random_crop_extra_max)
            s = max(0.0, ann.gt_start - float(l))
            e = ann.gt_end + float(r)
            if ann.query_time is not None:
                e = min(e, ann.query_time)
            if e <= ann.gt_end:
                e = ann.gt_end + 1.0
            return s, e
        raise ValueError(f"Unknown context_strategy: {self.context_strategy}")

    def _crop_keeping_gt(
        self, tokens: torch.Tensor, times: torch.Tensor, gt_start: float, gt_end: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = tokens.shape[0]
        if T <= self.max_seq_len:
            return tokens, times
        gt_idx = torch.where((times >= gt_start) & (times < gt_end))[0]
        if gt_idx.numel() == 0:
            s = random.randint(0, T - self.max_seq_len)
        else:
            gf = int(gt_idx.min())
            gl = int(gt_idx.max())
            mn = max(0, gl - self.max_seq_len + 1)
            mx = min(gf, T - self.max_seq_len)
            s = (
                random.randint(mn, mx)
                if mx >= mn
                else max(0, min(gf, T - self.max_seq_len))
            )
        return tokens[s : s + self.max_seq_len], times[s : s + self.max_seq_len]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ann, ws, we, win_name = self._get_ann_window(idx)
        try:
            gt_tok, _ = self.chunk_index.load_range(ann.video_id, ws, we)
            if gt_tok.numel() == 0:
                # fallback to exact GT if sampled window has no tokens on disk
                gt_tok, _ = self.chunk_index.load_range(
                    ann.video_id, ann.gt_start, ann.gt_end
                )
            if gt_tok.numel() == 0:
                raise RuntimeError(
                    f"Empty GT: video={ann.video_id} win=({ws:.1f},{we:.1f})"
                )
            target = self._resample_target(gt_tok)
            negs = (
                self._sample_temporal_negatives(ann, ws, we) if self.num_negs > 0 else []
            )

            cs, ce = self._context_range(ann)
            ctx, ctx_t = self.chunk_index.load_range(ann.video_id, cs, ce)
            if ctx.numel() == 0:
                ctx, ctx_t = self.chunk_index.load_range(ann.video_id, None, None)
            ctx, ctx_t = self._crop_keeping_gt(ctx, ctx_t, ann.gt_start, ann.gt_end)

        except Exception as e:
            print(
                f"[WARN] idx={idx} video={ann.video_id} win={win_name}: {e}. Fallback."
            )
            return self.__getitem__(random.randint(0, len(self) - 1))

        return {
            "context_tokens": ctx,
            "question": ann.question,
            "target_tokens": target,
            "target_len": int(target.shape[0]),
            "negatives": negs,
            "metadata": {
                "video_id": ann.video_id,
                "question": ann.question,
                "gt_start": ann.gt_start,
                "gt_end": ann.gt_end,
                "win_start": ws,
                "win_end": we,
                "window_name": win_name,
                "query_time": ann.query_time,
                "annotation_id": ann.annotation_id,
                "context_len": int(ctx.shape[0]),
            },
        }


def _pad_seq_batch(
    seqs: Sequence[torch.Tensor], dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad a list of ``[T_i, dim]`` tensors to ``[B, Tmax, dim]``.

    Returns ``(padded, mask [B, Tmax] bool, lengths [B] long)``.
    """
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    Tmax = int(lengths.max())
    out = torch.zeros(len(seqs), Tmax, dim)
    mask = torch.zeros(len(seqs), Tmax, dtype=torch.bool)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
        mask[i, : s.shape[0]] = True
    return out, mask, lengths


def build_collate_fn(tokenizer: CLIPTokenizer, max_question_len: int = 77):
    """Build a collate fn that pads context tokens, targets, negatives, and tokenizes questions.

    Targets are padded to the batch-max length (``target_mask`` / ``target_len``
    mark the real tokens). Negatives are padded over both the per-sample count
    ``K`` and the time axis, producing ``negatives [B, Kmax, Lmax, D]`` with
    ``negatives_mask [B, Kmax, Lmax]``. ``negatives`` is an empty tensor when no
    negatives were sampled.
    """

    def collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        dim = batch[0]["context_tokens"].shape[1]  # type: ignore

        # context
        ctx, cmsk, _ = _pad_seq_batch(
            [item["context_tokens"] for item in batch], dim  # type: ignore
        )

        # targets (variable length in length_aware mode)
        tgt, tmsk, tlen = _pad_seq_batch(
            [item["target_tokens"] for item in batch], dim  # type: ignore
        )

        # questions
        enc = tokenizer(
            [item["question"] for item in batch],  # type: ignore
            padding="max_length",
            truncation=True,
            max_length=max_question_len,
            return_tensors="pt",
        )

        # temporal negatives: pad over K (count) and L (time)
        neg_lists = [item["negatives"] for item in batch]  # type: ignore
        Kmax = max((len(n) for n in neg_lists), default=0)
        if Kmax > 0:
            Lmax = max(t.shape[0] for negs in neg_lists for t in negs)
            B = len(batch)
            negatives = torch.zeros(B, Kmax, Lmax, dim)
            neg_mask = torch.zeros(B, Kmax, Lmax, dtype=torch.bool)
            for i, negs in enumerate(neg_lists):
                for k, t in enumerate(negs):
                    negatives[i, k, : t.shape[0]] = t
                    neg_mask[i, k, : t.shape[0]] = True
        else:
            negatives = torch.zeros(len(batch), 0, 0, dim)
            neg_mask = torch.zeros(len(batch), 0, 0, dtype=torch.bool)

        return {
            "context_tokens": ctx,
            "context_mask": cmsk,
            "question_ids": enc["input_ids"],
            "question_mask": enc["attention_mask"].bool(),
            "target_tokens": tgt,
            "target_mask": tmsk,
            "target_len": tlen,
            "negatives": negatives,
            "negatives_mask": neg_mask,
            "metadata": [item["metadata"] for item in batch],
        }

    return collate
