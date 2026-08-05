"""Shared face-crop helpers used by inference and dataset prep."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CropConfig:
    """Padding and minimum size for face crops."""

    margin_ratio: float = 0.25
    min_side: int = 64


def expand_box(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    margin_ratio: float = 0.25,
) -> tuple[int, int, int, int]:
    """Expand a face box by margin_ratio, clamped to frame bounds."""
    height, width = frame_shape[:2]
    x, y, w, h = box
    pad_x = int(w * margin_ratio)
    pad_y = int(h * margin_ratio)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)
    return x0, y0, x1 - x0, y1 - y0


def crop_with_margin(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    config: CropConfig | None = None,
) -> np.ndarray | None:
    """Return a padded face crop, or None if the crop is too small."""
    cfg = config or CropConfig()
    x, y, w, h = expand_box(box, frame.shape, cfg.margin_ratio)
    if w < cfg.min_side or h < cfg.min_side:
        return None
    return frame[y : y + h, x : x + w]
