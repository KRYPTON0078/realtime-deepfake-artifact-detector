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

- FaceForensics++
- Celeb-DF v2

Document the exact subset, split, and preprocessing steps whenever you report results.

## Evaluation Caveats

- Scores are probabilistic, not legal proof
- Performance depends on face detection, lighting, and camera quality
- Temporal smoothing reduces flicker but adds slight delay

## Ethical Use

Use this project for research, education, and trustworthy-AI demonstrations. Pair automated scores with human review for high-stakes decisions.
