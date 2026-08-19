# Real-Time Deepfake Artifact Detector

A Flask demo that analyzes webcam, phone-camera frames, and uploaded video for **face-swap style manipulation artifacts** using OpenCV, PyTorch, and a CNN classifier.

> **Scope disclaimer:** This project detects spatial/blending artifacts common in classic face-swap pipelines. It does **not** claim general detection of all modern generative video models.

## Features

- Real-time webcam analysis with YuNet face detection and overlay
- Device-camera frame analysis for Android WebView / mobile browsers (`POST /analyze/frame`)
- Async uploaded-video jobs with pruning-safe queue APIs
- MobileNetV2 classifier with heuristic fallback
- Hysteresis temporal smoothing and calibrated thresholds
- Rich eval metrics (precision/recall/F1/AUC/ECE) and video-level scoring
- Android WebView wrapper with LAN cleartext support

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_demo_dataset.py
python training/train.py
python app/server.py
```

Open: `http://127.0.0.1:5000`

## Research-grade data

```bash
# Place licensed FF++ / Celeb-DF videos under data/raw_videos/{real,fake_face_swap}/
python scripts/download_sample_data.py --run-crop
python training/train.py
python training/evaluate.py
python scripts/calibrate_thresholds.py
```

## API Endpoints

- `GET /` dashboard
- `GET /about` limitations
- `GET /video_feed` MJPEG overlay stream (background worker)
- `GET /api/score` latest JSON score
- `GET/POST /api/config` runtime stride + thresholds
- `GET /health` mode, checkpoint, camera, worker status
- `POST /camera/start` / `POST /camera/stop`
- `POST /analyze/frame` base64 JPEG frame analysis
- `POST /analyze/upload` queue video job
- `GET /analyze/upload/<job_id>` job status
- `GET /analyze/upload/jobs` list jobs
- `DELETE /analyze/upload/<job_id>` delete job metadata

## Android

See `docs/ANDROID.md`. Emulator default URL is `http://10.0.2.2:5000/`. For a phone:

```bash
./gradlew assembleDebug -PBACKEND_URL=http://<LAN-IP>:5000/
```

## Tests

```bash
pytest tests/test_quality_gates.py -q
```

## Author

Magne Dina Neves

## Git identity (contributions)

Commits must use your verified GitHub email to count on your profile:

```bash
git config user.name "Magne Dina Neves"
git config user.email "magnedinanevesdina@gmail.com"
```

The Cursor auto-commit hook in `.cursor/hooks/` sets these automatically on Windows.
