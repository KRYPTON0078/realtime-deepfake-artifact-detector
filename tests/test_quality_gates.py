"""Unit tests for preprocess, labels, temporal smoother, and checkpoint loading."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.inference import DeepfakeAnalyzer, load_thresholds
from detector.preprocess import INPUT_SIZE, preprocess_face, preprocess_rgb_image
from detector.temporal import TemporalSmoother
from training.metrics import aggregate_video_scores, compute_binary_metrics


def test_preprocess_face_shape_and_dtype():
    image = np.zeros((120, 80, 3), dtype=np.uint8)
    tensor = preprocess_face(image)
    assert tuple(tensor.shape) == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype == __import__("torch").float32


def test_preprocess_rgb_matches_channel_layout():
    rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
    tensor = preprocess_rgb_image(rgb)
    assert tuple(tensor.shape) == (3, INPUT_SIZE, INPUT_SIZE)


def test_temporal_hysteresis_latches_fake_label():
    smoother = TemporalSmoother(window_size=3, enter_fake=0.6, exit_fake=0.4, warn_threshold=0.35)
    smoother.update(0.2)
    assert smoother.label_from_score() == "likely_real"
    smoother.update(0.9)
    smoother.update(0.9)
    smoother.update(0.9)
    assert smoother.label_from_score() == "likely_manipulated"
    smoother.update(0.5)
    # still latched until smoothed value falls below exit_fake
    assert smoother.label_from_score() == "likely_manipulated"
    smoother.update(0.1)
    smoother.update(0.1)
    smoother.update(0.1)
    assert smoother.label_from_score() == "likely_real"


def test_label_thresholds_on_analyzer():
    analyzer = DeepfakeAnalyzer(model_path=ROOT / "models" / "missing.pt")
    assert analyzer._label_from_score(0.9) == "likely_manipulated"
    assert analyzer._label_from_score(0.45) == "suspicious"
    assert analyzer._label_from_score(0.1) == "likely_real"


def test_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_binary_metrics(y_true, scores, threshold=0.5)
    assert metrics.accuracy == 1.0
    assert metrics.f1 == 1.0
    assert metrics.auc == 1.0


def test_video_aggregation():
    agg = aggregate_video_scores([0.1, 0.2, 0.9, 0.8, 0.3], top_k=2)
    assert agg["frames"] == 5
    assert agg["max"] == 0.9
    assert agg["topk_mean"] == pytest.approx(0.85)


def test_checkpoint_mode_smoke():
    model_path = ROOT / "models" / "artifact_detector.pt"
    if model_path.exists():
        analyzer = DeepfakeAnalyzer(model_path=model_path)
        assert analyzer.mode == "cnn"
    else:
        analyzer = DeepfakeAnalyzer(model_path=ROOT / "models" / "does-not-exist.pt")
        assert analyzer.mode == "heuristic"


def test_load_thresholds_defaults():
    thresholds = load_thresholds(ROOT / "models" / "no-such-thresholds.json")
    assert "fake_threshold" in thresholds
    assert thresholds["fake_threshold"] == 0.55
