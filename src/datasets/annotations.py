"""Annotation parsing.

Two consumers share this module:
    - extraction (:func:`load_videos_to_process`) needs the list of source
      videos to encode;
    - training (:func:`read_annotations`) needs the query / GT-moment records.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Training annotations: query + ground-truth moment span
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MomentAnnotation:
    video_id: str
    question: str
    gt_start: float
    gt_end: float
    query_time: Optional[float]
    annotation_id: Optional[int]


def read_annotations(path: str) -> List[MomentAnnotation]:
    """Parse the annotations JSON into :class:`MomentAnnotation` records.

    Accepts either a top-level list or a dict wrapping the list under
    ``annotations`` / ``data`` / ``samples``. Records lacking a video id,
    question, or a finite ``M1`` moment span are skipped.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        for key in ("annotations", "data", "samples"):
            if key in obj and isinstance(obj[key], list):
                obj = obj[key]
                break
    if not isinstance(obj, list):
        raise ValueError("Annotations must be a JSON list.")

    out: List[MomentAnnotation] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        vid = item.get("video_id")
        q = item.get("question")
        if not vid or not q:
            continue
        try:
            span = item["absolute_moments"]["M1"]["Moment_timespan"]
            gs = float(span["start_second"])
            ge = float(span["end_second"])
        except Exception:
            continue
        if not (math.isfinite(gs) and math.isfinite(ge)):
            continue
        qt = item.get("absolute_query_time")
        aid = item.get("annotation_id")
        out.append(
            MomentAnnotation(
                video_id=str(vid),
                question=str(q),
                gt_start=gs,
                gt_end=ge,
                query_time=float(qt) if qt is not None else None,
                annotation_id=int(aid) if aid is not None else None,
            )
        )
    if not out:
        raise ValueError(f"No valid annotations in {path}")
    return out


# ---------------------------------------------------------------------------
# Extraction: list of source videos to encode
# ---------------------------------------------------------------------------
def load_videos_to_process(
    annotations_path: str,
    dataset_filter: str = "ego4d",
    path_remap: Optional[dict] = None,
) -> List[dict]:
    """Return a deduplicated list of videos for the given dataset.

    Args:
        annotations_path: path to the annotations JSON.
        dataset_filter: only keep entries whose ``dataset`` matches (case-insensitive).
        path_remap: optional ``{"old_base": ..., "new_base": ...}`` mapping applied
            to each ``subfolder`` prefix (e.g. to translate cluster paths). When
            ``None`` no remapping is performed.

    Returns:
        List of ``{"dataset", "subfolder", "video_id"}`` dicts.
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    old_base = (path_remap or {}).get("old_base")
    new_base = (path_remap or {}).get("new_base")

    seen = set()
    videos: List[dict] = []
    for e in entries:
        dataset = str(e.get("dataset", "")).strip()
        if dataset.lower() != dataset_filter.lower():
            continue

        subfolder = e.get("subfolder")
        video_id = e.get("video_id")
        if not subfolder or not video_id:
            continue

        if old_base and new_base and subfolder.startswith(old_base):
            subfolder = subfolder.replace(old_base, new_base, 1)

        key = (dataset, subfolder, video_id)
        if key in seen:
            continue
        seen.add(key)

        videos.append(
            {"dataset": dataset, "subfolder": subfolder, "video_id": video_id}
        )
    return videos
