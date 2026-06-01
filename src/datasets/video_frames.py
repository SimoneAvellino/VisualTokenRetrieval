"""Video frame sampling and resizing for visual-token extraction.

Frames are decimated to a target FPS and grouped into fixed-size,
non-overlapping segments. The sampling rule
``skip = max(1, int(max(native_fps, 30.0) / target_fps))`` matches the
extraction behaviour used throughout the project.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import cv2
import numpy as np
from PIL import Image


def resize_video_frame(frame_bgr: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    """Convert a cv2 BGR ndarray to a resized RGB ``PIL.Image``.

    Args:
        frame_bgr: HxWx3 BGR frame from cv2.
        size: target ``(width, height)``.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized)


def iter_segments(
    video_path: str,
    target_fps: int = 1,
    window: int = 16,
    stride: int = 16,
) -> Iterator[Tuple[int, int, List[np.ndarray]]]:
    """Yield ``(start_second, end_second, frames)`` segments from a video.

    - ``frames`` is a list of ``window`` cv2 BGR ndarrays sampled at ``target_fps``.
    - ``start_second`` / ``end_second`` are inclusive (with target_fps=1, window=16
      the first segment covers [0, 15], the second [16, 31], ...).
    - Segments do not overlap when ``stride == window``.
    - Any trailing partial segment with fewer than ``window`` frames is dropped.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0:
            native_fps = 30.0
        skip = max(1, int(max(native_fps, 30.0) / target_fps))

        buffer: List[np.ndarray] = []
        raw_frame_idx = 0  # index in the source video stream
        sample_idx = 0  # index among the decimated samples
        next_emit_at = window  # emit when sample_idx reaches this

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if raw_frame_idx % skip == 0:
                buffer.append(frame)
                sample_idx += 1

                if sample_idx == next_emit_at:
                    start_s = sample_idx - window  # inclusive
                    end_s = sample_idx - 1  # inclusive
                    yield start_s, end_s, buffer
                    buffer = []  # stride == window => no overlap
                    next_emit_at += stride

            raw_frame_idx += 1
    finally:
        cap.release()
