# Dataset Layout

Place training images using this structure:

```text
data/
├── raw_videos/          # optional: source videos before cropping
│   ├── real/
│   └── fake_face_swap/
├── train/
│   ├── real/
│   └── fake_face_swap/
├── val/
│   ├── real/
│   └── fake_face_swap/
└── eval_clips/          # optional held-out videos for video-level eval
    ├── real/
    └── fake_face_swap/
```

## Quick demo dataset (synthetic)

```bash
python scripts/generate_demo_dataset.py
python training/train.py
```

Synthetic data is for plumbing checks only. Do not report synthetic accuracy as real deepfake performance.

## Research-grade path (FaceForensics++ / Celeb-DF)

1. Download licensed videos and organize under `data/raw_videos/{real,fake_face_swap}/`
2. Validate + crop:

```bash
python scripts/download_sample_data.py --run-crop
# or
python scripts/crop_faces_from_videos.py --source data/raw_videos --output data
```

3. Train / evaluate:

```bash
python training/train.py
python training/evaluate.py
```

Recommended subset for this project scope: FaceForensics++ Deepfakes + FaceSwap at **c23** compression. Document exact splits in `docs/LIMITATIONS.md`.

This project targets **face-swap spatial artifacts**, not all modern generative video models.
