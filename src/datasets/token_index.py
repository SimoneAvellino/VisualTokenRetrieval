"""On-disk index over extracted visual-token chunks.

Each ``.pt`` file holds the visual tokens for one ``<video_id>_<start>_<end>``
segment. :class:`TokenChunkIndex` maps filenames to absolute time ranges and
lazily loads / concatenates tokens for an arbitrary time window without holding
the whole dataset in RAM. Resolution-agnostic: it only handles ``[T, D]`` tensors.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def resample_tokens(tokens: torch.Tensor, out_len: int) -> torch.Tensor:
    """Resample ``[T, D]`` tokens to ``[out_len, D]`` via linear interpolation."""
    if tokens.ndim != 2:
        raise ValueError(f"Expected [T, D], got {tuple(tokens.shape)}")
    if tokens.shape[0] == 0:
        raise ValueError("Cannot resample empty tensor.")
    if tokens.shape[0] == out_len:
        return tokens.float()
    x = tokens.float().transpose(0, 1).unsqueeze(0)
    y = F.interpolate(x, size=out_len, mode="linear", align_corners=False)
    return y.squeeze(0).transpose(0, 1).contiguous()


@dataclass(frozen=True)
class TokenChunk:
    path: str
    video_id: str
    window_start: float
    window_end: float
    chunk_start_idx: int
    chunk_end_idx: int

    @property
    def abs_start(self) -> float:
        return self.window_start + float(self.chunk_start_idx)

    @property
    def abs_end_exclusive(self) -> float:
        return self.window_start + float(self.chunk_end_idx) + 1.0


def _parse_chunk_path(path: str) -> Optional[TokenChunk]:
    """Parse ``<video_id>_<start>_<end>.pt`` into a :class:`TokenChunk`."""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    vid, s, e = parts
    try:
        start = float(s)
        end = float(e)
        if end < start:
            return None
        return TokenChunk(
            path=path,
            video_id=vid,
            window_start=start,
            window_end=end,
            chunk_start_idx=0,
            chunk_end_idx=int(round(end - start)),
        )
    except ValueError:
        return None


class TokenChunkIndex:
    """Indexes all ``.pt`` chunk files without loading them into RAM."""

    def __init__(
        self, tokens_dir: str, recursive: bool = False, pool_to_fps: int = 1
    ) -> None:
        pattern = (
            os.path.join(tokens_dir, "**", "*.pt")
            if recursive
            else os.path.join(tokens_dir, "*.pt")
        )
        paths = sorted(glob.glob(pattern, recursive=recursive))
        if not paths:
            raise ValueError(f"No .pt files found in {tokens_dir}")
        self.pool_to_fps = int(pool_to_fps) if pool_to_fps else 0
        by_video: Dict[str, List[TokenChunk]] = {}
        skipped = 0
        for p in paths:
            c = _parse_chunk_path(p)
            if c is None:
                skipped += 1
                continue
            by_video.setdefault(c.video_id, []).append(c)
        for vid in by_video:
            by_video[vid].sort(key=lambda c: (c.abs_start, c.abs_end_exclusive))
        if not by_video:
            raise ValueError("No .pt files match <video_id>_<start>_<end>.pt")
        self.by_video = by_video
        self.num_files = len(paths)
        self.num_skipped = skipped

    def has_video(self, video_id: str) -> bool:
        return video_id in self.by_video

    def overlaps_range(self, video_id: str, start: float, end: float) -> bool:
        for c in self.by_video.get(video_id, []):
            if c.abs_start < end and c.abs_end_exclusive > start:
                return True
        return False

    @staticmethod
    def _load_tensor(path: str) -> torch.Tensor:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            for key in ("tokens", "visual_tokens", "features", "x", "embeddings"):
                if key in obj:
                    obj = obj[key]
                    break
        if not torch.is_tensor(obj):
            raise TypeError(f"{path}: does not contain a tensor.")
        t = obj.detach().cpu()
        if t.ndim == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        elif t.ndim > 2:
            t = t.reshape(-1, t.shape[-1])
        if t.ndim != 2:
            raise ValueError(f"{path}: expected [T,D], got {tuple(t.shape)}")
        return t.float()

    def load_range(
        self, video_id: str, start: Optional[float] = None, end: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and concatenate tokens overlapping ``[start, end)``.

        Returns ``(tokens [N, D], times [N])`` sorted by time with duplicate
        timestamps removed. Returns empty tensors when nothing matches.
        """
        if video_id not in self.by_video:
            raise KeyError(f"No chunks for video_id={video_id}")
        tl, tsl = [], []
        for c in self.by_video[video_id]:
            if start is not None and c.abs_end_exclusive <= start:
                continue
            if end is not None and c.abs_start >= end:
                continue
            tokens = self._load_tensor(c.path)
            if tokens.shape[0] == 0:
                continue
            dur = c.abs_end_exclusive - c.abs_start
            if self.pool_to_fps > 0:
                tgt = max(1, int(round(dur * self.pool_to_fps)))
                if tokens.shape[0] != tgt:
                    tokens = resample_tokens(tokens, tgt)
            L = tokens.shape[0]
            step = dur / float(L)
            times = c.abs_start + torch.arange(L, dtype=torch.float32) * step
            keep = torch.ones(L, dtype=torch.bool)
            if start is not None:
                keep &= times >= float(start)
            if end is not None:
                keep &= times < float(end)
            if keep.any():
                tl.append(tokens[keep])
                tsl.append(times[keep])
        if not tl:
            return torch.empty(0, 0), torch.empty(0)
        ta = torch.cat(tl)
        tsa = torch.cat(tsl)
        o = torch.argsort(tsa)
        ta = ta[o]
        tsa = tsa[o]
        if tsa.numel() > 1:
            r = torch.round(tsa * 1000) / 1000
            keep = torch.ones_like(r, dtype=torch.bool)
            keep[1:] = r[1:] != r[:-1]
            ta = ta[keep]
            tsa = tsa[keep]
        return ta, tsa
