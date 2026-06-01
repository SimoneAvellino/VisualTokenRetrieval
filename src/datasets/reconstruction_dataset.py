"""Supervised dataset + collate for query-conditioned GT token reconstruction.

Each item yields a context token sequence, a question string, and the target GT
tokens (resampled to a fixed length). Training uses :class:`GTWindowSampler`
augmentation; validation passes ``gt_sampler=None`` for exact GT windows.
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
    ) -> None:
        self.chunk_index = chunk_index
        self.out_len = out_len
        self.max_seq_len = max_seq_len
        self.context_strategy = context_strategy
        self.random_crop_extra_min = random_crop_extra_min
        self.random_crop_extra_max = random_crop_extra_max
        # None = no augmentation (val mode): always use exact GT
        self.gt_sampler = gt_sampler

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
        print(f"Dataset: {len(valid)} annotations | {skipped} skipped | mode={mode}")

    def __len__(self) -> int:
        return len(self.annotations)

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
            target = resample_tokens(gt_tok, self.out_len)

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


def build_collate_fn(tokenizer: CLIPTokenizer, max_question_len: int = 77):
    """Build a collate fn that pads context tokens and tokenizes questions."""

    def collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        lengths = [item["context_tokens"].shape[0] for item in batch]  # type: ignore
        ML = max(lengths)
        dim = batch[0]["context_tokens"].shape[1]  # type: ignore
        ctx = torch.zeros(len(batch), ML, dim)
        cmsk = torch.zeros(len(batch), ML, dtype=torch.bool)
        for i, item in enumerate(batch):
            x = item["context_tokens"]  # type: ignore
            ctx[i, : x.shape[0]] = x
            cmsk[i, : x.shape[0]] = True
        enc = tokenizer(
            [item["question"] for item in batch],  # type: ignore
            padding="max_length",
            truncation=True,
            max_length=max_question_len,
            return_tensors="pt",
        )
        return {
            "context_tokens": ctx,
            "context_mask": cmsk,
            "question_ids": enc["input_ids"],
            "question_mask": enc["attention_mask"].bool(),
            "target_tokens": torch.stack([item["target_tokens"] for item in batch]),  # type: ignore
            "metadata": [item["metadata"] for item in batch],
        }

    return collate
