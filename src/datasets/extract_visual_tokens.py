"""Entrypoint: extract Qwen3-VL visual tokens for every video segment.

For each non-overlapping ``window``-second segment of every selected video, the
projected vision embeddings are saved to::

    <output_path>/<dataset>/<video_id>_<start_second>_<end_second>.pt

with inclusive seconds (e.g. ``0_15``, ``16_31``, ...). All paths and parameters
come from a YAML config — there are no hardcoded paths.

Run::

    python -m src.datasets.extract_visual_tokens --config experiments/configs/extract_local.yaml
"""

from __future__ import annotations

import gc
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from ..utils.config import load_config, parse_config_arg
from .annotations import load_videos_to_process
from .qwen_encoder import encode_segment, load_model
from .video_frames import iter_segments, resize_video_frame


@dataclass
class ExtractConfig:
    """All extraction parameters (paths + sampling + model)."""

    # paths
    output_path: str
    annotations_path: str
    hf_cache_dir: str
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"

    # video selection
    dataset_filter: str = "ego4d"
    video_ext: str = ".mp4"
    # optional subfolder prefix remap, e.g. {"old_base": "...", "new_base": "..."}
    path_remap: Optional[dict] = None

    # frame sampling
    target_fps: int = 1
    window_size: int = 16
    stride: int = 16
    target_resolution: Tuple[int, int] = (448, 256)  # (W, H)

    # runtime
    attn_implementation: Optional[str] = "flash_attention_2"
    empty_cache_every: int = 32


def run(cfg: ExtractConfig) -> None:
    out_root = Path(cfg.output_path)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from {cfg.annotations_path}", flush=True)
    videos = load_videos_to_process(
        cfg.annotations_path,
        dataset_filter=cfg.dataset_filter,
        path_remap=cfg.path_remap,
    )
    print(f"Videos to process (after dedup): {len(videos)}", flush=True)

    print("Loading model…", flush=True)
    resolution = tuple(cfg.target_resolution)
    model, processor = load_model(
        cfg.model_id, cfg.hf_cache_dir, resolution, cfg.attn_implementation
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    first_logged = False
    segments_since_flush = 0
    total_saved = 0
    total_skipped_existing = 0
    total_failed_videos = 0

    for v in tqdm(videos, desc="videos"):
        dataset, subfolder, video_id = v["dataset"], v["subfolder"], v["video_id"]
        out_dir = out_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        video_path = f"{subfolder.rstrip('/')}/{video_id}{cfg.video_ext}"
        if not os.path.exists(video_path):
            print(f"[MISS] video not found: {video_path}", file=sys.stderr, flush=True)
            continue

        try:
            for start_s, end_s, frames_bgr in iter_segments(
                video_path,
                target_fps=cfg.target_fps,
                window=cfg.window_size,
                stride=cfg.stride,
            ):
                out_file = out_dir / f"{video_id}_{start_s}_{end_s}.pt"
                if out_file.exists():
                    total_skipped_existing += 1
                    continue

                pil_frames = [resize_video_frame(f, resolution) for f in frames_bgr]
                tokens = encode_segment(model, processor, pil_frames)

                if not first_logged:
                    num_tokens, hidden = tokens.shape
                    bytes_per_file = tokens.element_size() * tokens.numel()
                    print(
                        f"\n[INFO] First segment: shape={tuple(tokens.shape)} "
                        f"({num_tokens} tokens x {hidden} hidden) | "
                        f"~{bytes_per_file / 1024:.1f} KB per .pt file",
                        flush=True,
                    )
                    first_logged = True

                torch.save(tokens, out_file)
                total_saved += 1

                del pil_frames, tokens, frames_bgr
                segments_since_flush += 1
                if segments_since_flush >= cfg.empty_cache_every:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    segments_since_flush = 0

        except Exception as e:
            total_failed_videos += 1
            print(f"[ERR ] {video_path}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

    print("=" * 60, flush=True)
    print(f"Resolution used        : {resolution} (W x H)", flush=True)
    print(f"Saved segments         : {total_saved}", flush=True)
    print(f"Skipped (already exist): {total_skipped_existing}", flush=True)
    print(f"Failed videos          : {total_failed_videos}", flush=True)


def main() -> None:
    config_path = parse_config_arg("Extract Qwen3-VL visual tokens from videos.")
    cfg = load_config(ExtractConfig, config_path)
    run(cfg)


if __name__ == "__main__":
    main()
