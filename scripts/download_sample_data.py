"""Prepare research-grade face-swap training data from public datasets.

FaceForensics++ and Celeb-DF require accepting their licenses and downloading
manually. This script documents the expected layout and can:

1. Print download / license instructions
2. Validate a local raw_videos tree
3. Invoke the face-crop pipeline into data/train and data/val

Recommended subsets for this project's scope (face-swap spatial artifacts):

- FaceForensics++: Deepfakes + FaceSwap manipulations, c23 compression preferred
- Celeb-DF v2: YouTube-real vs Celeb-synthesis fakes (harder, optional holdout)

Document the exact subset, split seed, and compression level in docs/LIMITATIONS.md
whenever you report results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INSTRUCTIONS = """
Face-swap artifact detector — research dataset setup
====================================================

Quick local demo (synthetic, not for reported accuracy):
    python scripts/generate_demo_dataset.py
    python training/train.py

Research-grade path
-------------------
1. Accept licenses and download:
   - FaceForensics++: https://github.com/ondyari/FaceForensics
   - Celeb-DF v2: https://github.com/yuezunli/celeb-deepfakeforensics

2. Organize videos as:
    data/raw_videos/real/*.mp4
    data/raw_videos/fake_face_swap/*.mp4

   Suggested mapping:
   - FF++ original sequences          -> real/
   - FF++ Deepfakes + FaceSwap (c23)  -> fake_face_swap/
   - Optional Celeb-DF holdout        -> data/eval_clips/ (see scripts/eval_videos.py)

3. Crop faces with the same detector/padding used at inference:
    python scripts/crop_faces_from_videos.py --source data/raw_videos --output data

4. Train and evaluate:
    python training/train.py
    python training/evaluate.py

Keep scope limited to face-swap spatial artifacts. Do not claim general deepfake detection.
"""


def validate_raw_tree(source: Path) -> dict:
    report = {"ok": True, "classes": {}}
    for label in ("real", "fake_face_swap"):
        class_dir = source / label
        videos = []
        if class_dir.exists():
            videos = [
                path.name
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            ]
        report["classes"][label] = {"exists": class_dir.exists(), "videos": len(videos)}
        if len(videos) == 0:
            report["ok"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset prep helper for FF++ / Celeb-DF")
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "raw_videos")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--run-crop",
        action="store_true",
        help="After validating the raw tree, run scripts/crop_faces_from_videos.py",
    )
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-faces-per-video", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(INSTRUCTIONS)
    report = validate_raw_tree(args.source)
    print("Validation:")
    print(json.dumps(report, indent=2))

    manifest_path = args.output / "dataset_manifest.json"
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(args.source),
        "recommended_subsets": {
            "FaceForensics++": ["Deepfakes", "FaceSwap"],
            "compression": "c23",
            "optional_holdout": "Celeb-DF v2",
        },
        "validation": report,
        "split_seed": args.seed,
        "val_ratio": args.val_ratio,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")

    if not args.run_crop:
        return
    if not report["ok"]:
        raise SystemExit("Cannot run crop: raw_videos tree is incomplete.")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "crop_faces_from_videos.py"),
        "--source",
        str(args.source),
        "--output",
        str(args.output),
        "--frame-stride",
        str(args.frame_stride),
        "--max-faces-per-video",
        str(args.max_faces_per_video),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
