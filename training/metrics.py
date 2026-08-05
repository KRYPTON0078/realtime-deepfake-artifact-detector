"""Classification and calibration metrics for artifact detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    ece: float | None = None
    threshold: float = 0.5
    support: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(y_true)) < 2:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    sum_ranks = float(ranks[pos].sum())
    return (sum_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def expected_calibration_error(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(scores) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        mask = (scores >= bins[index]) & (scores < bins[index + 1] if index < n_bins - 1 else scores <= bins[index + 1])
        if not np.any(mask):
            continue
        conf = float(scores[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (mask.sum() / len(scores)) * abs(acc - conf)
    return float(ece)


def compute_binary_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> BinaryMetrics:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    y_pred = (scores >= threshold).astype(int)
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=roc_auc(y_true, scores),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        ece=expected_calibration_error(y_true, scores),
        threshold=threshold,
        support=int(len(y_true)),
    )


def best_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, BinaryMetrics]:
    """Scan thresholds and return the F1-maximizing operating point."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    candidates = np.unique(np.concatenate([scores, np.array([0.0, 0.5, 1.0])]))
    best_threshold = 0.5
    best_metrics = compute_binary_metrics(y_true, scores, threshold=0.5)
    for threshold in candidates:
        metrics = compute_binary_metrics(y_true, scores, threshold=float(threshold))
        if metrics.f1 > best_metrics.f1:
            best_metrics = metrics
            best_threshold = float(threshold)
    return best_threshold, best_metrics


def aggregate_video_scores(frame_scores: list[float], top_k: int = 5) -> dict:
    """Aggregate frame-level fake probabilities into a video-level summary."""
    if not frame_scores:
        return {
            "frames": 0,
            "mean": 0.0,
            "median": 0.0,
            "topk_mean": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(frame_scores, dtype=float)
    k = max(1, min(top_k, len(arr)))
    topk = np.sort(arr)[-k:]
    return {
        "frames": int(len(arr)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "topk_mean": float(topk.mean()),
        "max": float(arr.max()),
    }
