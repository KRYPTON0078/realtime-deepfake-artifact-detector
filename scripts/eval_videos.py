"""Video-level evaluation on held-out clips under data/eval_clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.inference import DeepfakeAnalyzer
from training.metrics import aggregate_video_scores, compute_binary_metrics

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def iter_labeled_videos(root: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    mapping = {"real": 0, "fake_face_swap": 1}
    for label_name, label in mapping.items():
        class_dir = root / label_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                samples.append((path, label))
    return samples


def score_video(analyzer: DeepfakeAnalyzer, video_path: Path, frame_stride: int) -> list[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    scores: list[float] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_stride != 0:
            frame_index += 1
            continue
        result = analyzer.analyze_frame(frame, smooth=False)
        if result.face_detected:
            scores.append(result.fake_probability)
        frame_index += 1
    cap.release()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate video-level deepfake scores")
    parser.add_argument("--clips", type=Path, default=ROOT / "data" / "eval_clips")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "models" / "video_eval_metrics.json",
    )
    args = parser.parse_args()

    samples = iter_labeled_videos(args.clips)
    if not samples:
        raise SystemExit(
            f"No eval clips found under {args.clips}. "
            "Place videos in eval_clips/real and eval_clips/fake_face_swap."
        )

    analyzer = DeepfakeAnalyzer()
    video_labels: list[int] = []
    video_scores: list[float] = []
    details = []
    for path, label in samples:
        frame_scores = score_video(analyzer, path, args.frame_stride)
        agg = aggregate_video_scores(frame_scores, top_k=args.top_k)
        video_labels.append(label)
        video_scores.append(agg["topk_mean"])
        details.append(
            {
                "path": str(path),
                "label": label,
                "aggregation": agg,
            }
        )

    metrics = compute_binary_metrics(
        np.asarray(video_labels),
        np.asarray(video_scores),
        threshold=args.threshold,
    )
    report = {
        "clips_root": str(args.clips),
        "mode": analyzer.mode,
        "threshold": args.threshold,
        "metrics": metrics.to_dict(),
        "videos": details,
    }
    print(json.dumps(report, indent=2))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
