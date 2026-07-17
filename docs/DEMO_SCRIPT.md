# 2-Minute Demo Script

## Setup

1. Start the app:
   ```bash
   python app/server.py
   ```
2. Open `http://127.0.0.1:5000`

## Flow

### 1. Explain the scope (20 seconds)

Say:

> This detector focuses on face-swap spatial artifacts. It is not a universal deepfake detector for all modern generative video models.

### 2. Live webcam demo (40 seconds)

1. Click **Start Camera**
2. Show your real face and note a lower fake probability
3. Point out the face box, label, and confidence bar
4. Mention whether the app is in `heuristic` or `cnn` mode

### 3. Uploaded manipulated clip (40 seconds)

1. Upload a short manipulated or heavily compressed face clip
2. Show average fake probability and manipulated frame ratio
3. Explain that the score rises when spatial artifacts are present

### 4. Limitations close (20 seconds)

Say:

> Newer diffusion-based fakes may evade this model because they do not always leave the same blending artifacts as classic face-swap pipelines. That is why we frame this as an artifact detector for a specific attack class.

## Optional Training Demo

```bash
python scripts/generate_demo_dataset.py
python training/train.py
```

Restart the app to switch from heuristic mode to CNN mode.
