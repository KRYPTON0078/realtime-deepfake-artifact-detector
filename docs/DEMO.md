# Demo video (placeholder)

This repo ships a **live dashboard**, not a baked-in recording. Drop a 2–3 minute walkthrough here when you record it so recruiters and reviewers can watch without installing anything.

Spoken talking points live in [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) (~2 minutes).

## Where to put the video

Pick one (both is fine):

1. **GitHub Release asset** on `v0.1.0` (preferred): attach `deepfake-artifact-detector-demo.mp4`
2. **This folder**: `docs/demo/deepfake-artifact-detector-demo.mp4` (keep it out of git if the file is large; link it from the README instead)
3. **External link**: unlisted YouTube / Google Drive / university hosting

Then replace the placeholder in the README **Demo video** section with the URL.

Suggested README snippet after you have a link:

```markdown
## Demo video

[2–3 min walkthrough](https://YOUR-LINK-HERE) — webcam overlay, upload job, scope disclaimer.
```

## What to record (2–3 minutes)

| Time | Show | Say |
| --- | --- | --- |
| 0:00–0:20 | README / `/about` disclaimer | This scores **face-swap spatial artifacts**, not every modern generative video model. |
| 0:20–1:10 | Dashboard → **Start Camera** | Face box, label, fake probability, `heuristic` vs `cnn` mode. Real face should stay lower-risk if lighting is decent. |
| 1:10–2:10 | **Upload Video** of a short manipulated or heavily compressed face clip | Average fake probability + manipulated frame ratio. Scores rise when blending / smoothing artifacts are visible. |
| 2:10–2:40 | `/health` JSON or About page | Limitations: no FaceForensics++ numbers in-repo; synthetic val accuracy is plumbing only. |

Optional extra (if you trained a checkpoint): restart after `python training/train.py` and show `mode: cnn` in `GET /health`.

## Recording tips

- 1280×720 or 1920×1080, 2–3 minutes, no background music
- Show the terminal once: `./scripts/demo_up.sh` then the browser at `http://127.0.0.1:5000`
- Keep the on-screen disclaimer readable
- Do not imply legal proof or “universal deepfake detection”

## Checklist before attaching to the release

- [ ] File named `deepfake-artifact-detector-demo.mp4` (or a stable URL)
- [ ] README **Demo video** section points at the asset
- [ ] Release notes in `docs/RELEASE_v0.1.0.md` mention the video
- [ ] No copyrighted movie clips without a license — use your own face + a licensed/self-made swap, or a dataset clip you are allowed to show
