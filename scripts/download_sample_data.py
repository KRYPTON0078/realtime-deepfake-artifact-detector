"""Dataset download instructions for FaceForensics++ style training."""

from __future__ import annotations

print(
    """
Face-swap artifact detector dataset setup
=======================================

For a quick local demo, run:
    python scripts/generate_demo_dataset.py

For research-grade training, prepare cropped face images in:
    data/train/real/
    data/train/fake_face_swap/
    data/val/real/
    data/val/fake_face_swap/

Recommended public datasets:
- FaceForensics++ (face-swap / Deepfakes subsets)
- Celeb-DF (v2)

Keep the demo scope limited to face-swap spatial artifacts.
Document the exact subset used in docs/LIMITATIONS.md.
"""
)
