"""Temporal smoothing for frame-level fake scores."""

from __future__ import annotations

from collections import deque


class TemporalSmoother:
    """Rolling smoother with optional confidence weighting and hysteresis labels."""

    def __init__(
        self,
        window_size: int = 8,
        enter_fake: float = 0.60,
        exit_fake: float = 0.45,
        warn_threshold: float = 0.40,
        use_confidence_weights: bool = True,
    ) -> None:
        self.window_size = window_size
        self.enter_fake = enter_fake
        self.exit_fake = exit_fake
        self.warn_threshold = warn_threshold
        self.use_confidence_weights = use_confidence_weights
        self._scores: deque[float] = deque(maxlen=window_size)
        self._weights: deque[float] = deque(maxlen=window_size)
        self._latched_fake = False

    def update(self, score: float, confidence: float = 1.0) -> float:
        weight = max(0.05, float(confidence)) if self.use_confidence_weights else 1.0
        self._scores.append(float(score))
        self._weights.append(weight)
        smoothed = self.value
        if smoothed >= self.enter_fake:
            self._latched_fake = True
        elif smoothed <= self.exit_fake:
            self._latched_fake = False
        return smoothed

    @property
    def value(self) -> float:
        if not self._scores:
            return 0.0
        total_weight = sum(self._weights) or 1.0
        return sum(score * weight for score, weight in zip(self._scores, self._weights)) / total_weight

    def label_from_score(self, score: float | None = None) -> str:
        value = self.value if score is None else float(score)
        if self._latched_fake or value >= self.enter_fake:
            return "likely_manipulated"
        if value >= self.warn_threshold:
            return "suspicious"
        return "likely_real"

    def reset(self) -> None:
        self._scores.clear()
        self._weights.clear()
        self._latched_fake = False
