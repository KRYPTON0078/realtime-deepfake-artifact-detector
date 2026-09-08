# Release notes — v0.1.0

**Tag:** `v0.1.0`  
**Title:** Real-Time Deepfake Artifact Detector v0.1.0  
**Author:** Magne Dina Neves ([KRYPTON0078](https://github.com/KRYPTON0078)), University of Macau

Paste this body into the GitHub Release after merging to `main` and pushing the tag.

```markdown
First recruiter/professor-ready cut of the face-swap **artifact** detector.

This release is a scoped research demo: YuNet + MobileNetV2 (or heuristics), a Flask dashboard, and JSON APIs. It does **not** claim general detection of all modern generative video models.

## Try it

```bash
git clone https://github.com/KRYPTON0078/realtime-deepfake-artifact-detector.git
cd realtime-deepfake-artifact-detector
./scripts/demo_up.sh
```

Open http://127.0.0.1:5000 — webcam overlay, device-camera frames, video upload.

Health check: `GET /health` (`mode` is `heuristic` until you train `models/artifact_detector.pt`).

Windows: `scripts\demo_up.bat`

## What is in scope

- Classic face-swap spatial artifacts (blend boundaries, over-smoothing, color mismatch, compression)
- Live scoring with hysteresis so labels do not flicker every frame
- Upload jobs with average fake probability and manipulated-frame ratio

## What is not in this release

- Benchmark numbers on FaceForensics++ or Celeb-DF (protocol is documented; licensed videos are not bundled)
- A shipped CNN checkpoint (too large / training-data dependent; train locally)
- A demo MP4 — drop one later using `docs/DEMO.md` and attach it to this release

## Honest eval

`models/calibrated_thresholds.json` records **synthetic** validation metrics (n=24, accuracy/F1/AUC = 1.0). That is a plumbing check on cartoon face crops, not real-world deepfake performance. See `docs/TECH_REPORT.md`.

## Docs

- README — problem, one-command demo, architecture, results
- `docs/TECH_REPORT.md` — method, pipeline, eval protocol, limits
- `docs/LIMITATIONS.md` — attack class and ethics
- `docs/DEMO.md` / `docs/DEMO_SCRIPT.md` — video placeholder and 2-minute talk track
- `CHANGELOG.md`

## Suggested GitHub topics

`deepfake-detection` `computer-vision` `media-forensics` `pytorch` `opencv` `flask` `mobilenetv2` `face-swap` `trustworthy-ai` `yunet`
```

## Publish checklist (after merge)

1. [ ] PR merged to `main`
2. [ ] `git checkout main && git pull`
3. [ ] `git tag -a v0.1.0 -m "v0.1.0: recruiter/professor demo cut"`
4. [ ] `git push origin v0.1.0`
5. [ ] GitHub → Releases → Draft a new release from `v0.1.0`
6. [ ] Paste the markdown body above
7. [ ] Optional: attach demo MP4 when Magne records it (`docs/DEMO.md`)
8. [ ] Settings → Topics: add the keywords listed above
9. [ ] Confirm `LICENSE` (MIT) is visible on the repo home page
