"""Temporal smoothing for frame-level fake scores."""

from __future__ import annotations

from collections import deque


class TemporalSmoother:
    """Rolling average smoother to reduce score flicker."""

    def __init__(self, window_size: int = 8) -> None:
        self.window_size = window_size
        self._scores: deque[float] = deque(maxlen=window_size)

    def update(self, score: float) -> float:
        self._scores.append(float(score))
        return sum(self._scores) / len(self._scores)

    def reset(self) -> None:
        self._scores.clear()
