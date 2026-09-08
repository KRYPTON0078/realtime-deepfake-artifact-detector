# Changelog

All notable changes to this project are documented here.

## [0.1.0] — 2026-09-08

First public portfolio / course-demo cut: a scoped face-swap **artifact** detector with a Flask dashboard, JSON APIs, and an honest evaluation story.

### Added

- One-command demo: `scripts/demo_up.sh` (Unix) and `scripts/demo_up.bat` (Windows); Unix twin `run.sh` for `run.bat`
- `GET /health` smoke coverage in `tests/test_health.py`
- Tech writeup: `docs/TECH_REPORT.md`
- Demo-video drop instructions: `docs/DEMO.md`
- MIT `LICENSE`, `CITATION.cff`, and GitHub Release notes in `docs/RELEASE_v0.1.0.md`

### Documented (already in the codebase)

- YuNet face detection with Haar fallback, MobileNetV2 classifier, heuristic fallback
- Webcam MJPEG overlay, mobile `POST /analyze/frame`, async video upload jobs
- Hysteresis temporal smoothing and calibrated thresholds
- Image-level metrics (precision / recall / F1 / AUC / ECE) and video-level aggregation
- Android WebView wrapper

### Honesty notes

- Checked-in numbers in `models/calibrated_thresholds.json` are **synthetic demo** metrics (n=24). They are not FaceForensics++ or Celeb-DF results.
- No trained `models/artifact_detector.pt` is shipped; default runtime mode is `heuristic` until you train locally.
