"""Random ground-truth window sampler (data augmentation).

At each sample, a window that always contains ``[gt_start, gt_end]`` is drawn by
padding both sides by independent uniform amounts. Over many epochs the model
sees a diverse range of contexts around the same moment. With probability
``exact_prob`` the clean GT window is returned instead, so the model still sees
unpadded targets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class GTWindowSampler:
    """Samples a random window that always contains ``[gt_start, gt_end]``.

    Args:
        max_left_pad:  max seconds sampled before ``gt_start`` (default 30s).
        max_right_pad: max seconds sampled after ``gt_end``    (default 30s).
        zero_dur_pad:  padding applied to zero-duration moments (default 5s).
        exact_prob:    probability of returning the exact GT window (default 0.2).
    """

    max_left_pad: float = 30.0
    max_right_pad: float = 30.0
    zero_dur_pad: float = 5.0
    exact_prob: float = 0.2

    def sample(
        self,
        gt_start: float,
        gt_end: float,
        video_end: Optional[float] = None,
    ) -> Tuple[float, float, str]:
        """Return ``(window_start, window_end, name)``.

        The window always satisfies ``window_start <= gt_start`` and
        ``window_end >= gt_end``.
        """
        # handle zero-duration moments
        if gt_end <= gt_start:
            gs = gt_start - self.zero_dur_pad
            ge = gt_end + self.zero_dur_pad
        else:
            gs, ge = gt_start, gt_end

        # with exact_prob return the clean GT window (no padding)
        if random.random() < self.exact_prob:
            return max(0.0, gs), ge, "gt_exact"

        # sample left and right pads independently
        max_l = min(gs, self.max_left_pad)
        max_r = self.max_right_pad
        if video_end is not None:
            max_r = min(max_r, video_end - ge)
        max_r = max(0.0, max_r)

        left_pad = random.uniform(0.0, max_l)
        right_pad = random.uniform(0.0, max_r)

        ws = max(0.0, gs - left_pad)
        we = ge + right_pad
        we = max(ws + 1.0, we)

        name = f"rand_l{left_pad:.1f}_r{right_pad:.1f}"
        return ws, we, name
