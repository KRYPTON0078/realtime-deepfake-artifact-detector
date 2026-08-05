# Scope and Limitations

## Intended Attack Class

This project targets **face-swap style spatial artifacts**, including:

- blending boundaries around the jaw and cheeks
- over-smoothing or GAN-like texture loss
- color inconsistency between face and background
- compression and re-encoding artifacts

## Out of Scope

The demo does **not** claim reliable detection for:

- diffusion-based video generation
- high-quality neural rendering pipelines
- audio-only or full-body deepfakes without visible facial artifacts
- all unseen generative models

## Model Modes

- **heuristic**: used when `models/artifact_detector.pt` is missing; based on simple spatial cues
- **cnn**: MobileNetV2 checkpoint trained on `real` vs `fake_face_swap` folders

## Dataset Guidance

Recommended public datasets for stronger training:

- FaceForensics++ (Deepfakes + FaceSwap manipulations; prefer **c23** compression)
- Celeb-DF v2 (harder holdout / cross-dataset check)

Prep pipeline:

```bash
# organize licensed videos under data/raw_videos/{real,fake_face_swap}/
python scripts/download_sample_data.py --run-crop
python training/train.py
python training/evaluate.py
```

Face crops use the project detector with margin padding (`scripts/crop_faces_from_videos.py`) so train and inference share the same crop policy.

Document the exact subset, split seed, compression level, and crop settings whenever you report results. The bundled synthetic demo dataset is for local plumbing only — a perfect synthetic val accuracy is not evidence of real-world deepfake detection quality.

## Evaluation Caveats

- Scores are probabilistic, not legal proof
- Performance depends on face detection, lighting, and camera quality
- Temporal smoothing reduces flicker but adds slight delay

## Ethical Use

Use this project for research, education, and trustworthy-AI demonstrations. Pair automated scores with human review for high-stakes decisions.
