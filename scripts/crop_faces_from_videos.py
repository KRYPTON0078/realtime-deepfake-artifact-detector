"""Extract padded face crops from labeled video folders for training.

Expected input layout (FaceForensics++ / Celeb-DF style after manual download):

    raw_videos/
    ├── real/*.mp4
    └── fake_face_swap/*.mp4

Output layout:

    data/train/{real,fake_face_swap}/*.jpg
    data/val/{real,fake_face_swap}/*.jpg
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.face_crop import CropConfig, crop_with_margin
from detector.face_detector import FaceDetector

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def iter_videos(class_dir: Path) -> list[Path]:
    if not class_dir.exists():
        return []
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def extract_faces_from_video(
    video_path: Path,
    detector: FaceDetector,
    crop_config: CropConfig,
    frame_stride: int,
    max_faces: int,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    crops: list[np.ndarray] = []
    frame_index = 0
    while len(crops) < max_faces:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_stride != 0:
            frame_index += 1
            continue
        face_crop, detection = detector.crop_largest_face(frame)
        if detection is not None:
            padded = crop_with_margin(frame, detection.box, crop_config)
            if padded is not None:
                crops.append(padded)
        frame_index += 1
    cap.release()
    return crops


def write_crops(
    crops: list,
    output_dir: Path,
    prefix: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, crop in enumerate(crops):
        path = output_dir / f"{prefix}_{index:05d}.jpg"
        if cv2.imwrite(str(path), crop):
            written += 1
    return written


def prepare_split(
    source_root: Path,
    output_root: Path,
    val_ratio: float,
    frame_stride: int,
    max_faces_per_video: int,
    margin_ratio: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    detector = FaceDetector()
    crop_config = CropConfig(margin_ratio=margin_ratio)
    summary: dict = {"train": {}, "val": {}}

    for label in ("real", "fake_face_swap"):
        videos = iter_videos(source_root / label)
        rng.shuffle(videos)
        split_at = max(1, int(len(videos) * (1.0 - val_ratio))) if videos else 0
        train_videos = videos[:split_at]
        val_videos = videos[split_at:] if len(videos) > 1 else []

        for split_name, split_videos in (("train", train_videos), ("val", val_videos)):
            out_dir = output_root / split_name / label
            total = 0
            for video_path in split_videos:
                crops = extract_faces_from_video(
                    video_path=video_path,
                    detector=detector,
                    crop_config=crop_config,
                    frame_stride=frame_stride,
                    max_faces=max_faces_per_video,
                )
                total += write_crops(crops, out_dir, prefix=f"{video_path.stem}_{label}")
            summary[split_name][label] = {"videos": len(split_videos), "crops": total}

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop training faces from labeled videos")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "raw_videos",
        help="Root with real/ and fake_face_swap/ video folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data",
        help="Dataset root containing train/ and val/",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-faces-per-video", type=int, default=30)
    parser.add_argument("--margin-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(
            f"Source folder not found: {args.source}\n"
            "Download FaceForensics++ / Celeb-DF videos and place them under "
            "data/raw_videos/{real,fake_face_swap}/ then re-run."
        )

    summary = prepare_split(
        source_root=args.source,
        output_root=args.output,
        val_ratio=args.val_ratio,
        frame_stride=args.frame_stride,
        max_faces_per_video=args.max_faces_per_video,
        margin_ratio=args.margin_ratio,
        seed=args.seed,
    )
    print("Face crop summary:")
    for split_name, labels in summary.items():
        for label, stats in labels.items():
            print(f"  {split_name}/{label}: {stats['videos']} videos -> {stats['crops']} crops")


if __name__ == "__main__":
    main()
