# Real-Time Deepfake Artifact Detector

A Flask demo app that analyzes webcam and uploaded video feeds for **face-swap style manipulation artifacts** using OpenCV, PyTorch, and a CNN classifier.

> **Scope disclaimer:** This project detects spatial/blending artifacts common in classic face-swap pipelines. It does **not** claim general detection of all modern generative video models.

## Features

- Real-time webcam analysis with face bounding boxes and fake-probability overlay
- Device-camera frame analysis endpoint for mobile/WebView live scoring
- Uploaded video analysis with sampled-frame scoring
- Upload job queue listing and retention cleanup controls
- Frame-analysis concurrency guard to avoid overload on mobile polling
- PyTorch MobileNetV2 binary classifier (`real` vs `fake_face_swap`)
- Heuristic fallback mode when no trained checkpoint is available
- Temporal smoothing to reduce score flicker
- Honest limitations page for demo and portfolio use

## Tech Stack

- Python
- PyTorch / torchvision
- OpenCV
- Flask

## Project Structure

```text
DeepfakeDetector/
├── app/                 # Flask UI and API
├── detector/            # face detection, preprocessing, inference
├── training/            # dataset + train/eval scripts
├── scripts/             # demo dataset generator
├── models/              # saved checkpoints
└── docs/                # limitations + demo script
```

## Quick Start

```bash
cd D:\DeepfakeDetector
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app/server.py
```

Open: `http://127.0.0.1:5000`

## Train A Demo Model

```bash
python scripts/generate_demo_dataset.py
python training/train.py
python training/evaluate.py
```

This creates `models/artifact_detector.pt` and switches the app to CNN mode on restart.

## API Endpoints

- `GET /` dashboard
- `GET /about` limitations
- `GET /video_feed` MJPEG stream with overlay
- `GET /api/score` latest JSON score
- `GET /api/config` runtime config and model mode
- `GET /health` service health
- `POST /camera/start` start webcam
- `POST /camera/stop` stop webcam
- `POST /analyze/frame` analyze a base64 JPEG frame payload
- `POST /analyze/upload` queue uploaded video analysis job
- `GET /analyze/upload/<job_id>` fetch upload job status/result
- `GET /analyze/upload/jobs` list upload jobs
- `DELETE /analyze/upload/<job_id>` delete upload job metadata

## Android Wrapper

A minimal Android WebView wrapper project is available in `android/`.

- See `docs/ANDROID.md` for setup and LAN connection details.
- The Android app wraps the existing Flask UI and upload flow.

## Research Relevance

This project supports digital trust and content verification research by showing:

- how CNN-based artifact detectors can flag a specific manipulation class
- why model scope must be stated clearly
- how real-time media analysis pipelines combine vision, ML, and web interfaces

## Author

Magne Dina Neves

## License

MIT (add license file if publishing publicly)
