# Held-out evaluation clips

Place short videos here for video-level metrics (`scripts/eval_videos.py`):

```text
data/eval_clips/
├── real/*.mp4
└── fake_face_swap/*.mp4
```

Suggested use: keep FaceForensics++ c23 Deepfakes/FaceSwap clips out of the train/val crop set, or use Celeb-DF v2 as a cross-dataset holdout.

```bash
python scripts/eval_videos.py --clips data/eval_clips --output-json models/video_eval_metrics.json
```
