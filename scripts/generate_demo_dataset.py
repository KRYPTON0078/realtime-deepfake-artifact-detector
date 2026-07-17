"""Generate a small synthetic demo dataset for local training."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _write_face_like_image(path: Path, seed: int, manipulated: bool) -> None:
    rng = np.random.default_rng(seed)
    image = rng.integers(40, 180, size=(224, 224, 3), dtype=np.uint8)

    center = (112, 112)
    axes = (70, 90)
    cv2.ellipse(image, center, axes, 0, 0, 360, (220, 190, 170), -1)
    cv2.circle(image, (85, 95), 8, (20, 20, 20), -1)
    cv2.circle(image, (139, 95), 8, (20, 20, 20), -1)
    cv2.ellipse(image, (112, 140), (25, 12), 0, 0, 180, (40, 20, 20), 2)

    if manipulated:
        cv2.GaussianBlur(image, (9, 9), 0, dst=image)
        noise = rng.integers(-25, 25, size=image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.rectangle(image, (60, 70), (164, 170), (255, 0, 0), 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def generate_dataset(output_root: Path, samples_per_class: int = 40) -> None:
    for split in ("train", "val"):
        for label in ("real", "fake_face_swap"):
            manipulated = label == "fake_face_swap"
            count = samples_per_class if split == "train" else max(8, samples_per_class // 5)
            for index in range(count):
                seed = hash((split, label, index)) % (2**32)
                path = output_root / split / label / f"{label}_{index:03d}.jpg"
                _write_face_like_image(path, seed, manipulated=manipulated)
    print(f"Generated demo dataset under {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
    )
    parser.add_argument("--samples", type=int, default=40)
    args = parser.parse_args()
    generate_dataset(args.output, args.samples)
