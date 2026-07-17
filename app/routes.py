"""Flask routes for the deepfake artifact detector demo."""

from __future__ import annotations

import base64
import binascii
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, Response, jsonify, render_template, request

from app.state import UPLOAD_DIR, state

bp = Blueprint("main", __name__)
UPLOAD_WORKERS = max(1, int(os.getenv("UPLOAD_ANALYSIS_WORKERS", "2")))
upload_executor = ThreadPoolExecutor(max_workers=UPLOAD_WORKERS)
upload_jobs: dict[str, dict] = {}
upload_jobs_lock = threading.Lock()
UPLOAD_JOB_RETENTION_SECONDS = max(60, int(os.getenv("UPLOAD_JOB_RETENTION_SECONDS", "1800")))
MAX_UPLOAD_JOBS = max(10, int(os.getenv("MAX_UPLOAD_JOBS", "200")))
FRAME_MAX_DIM = max(160, int(os.getenv("FRAME_MAX_DIM", "640")))

DISCLAIMER = (
    "Detector trained for face-swap artifact patterns; "
    "not validated against all modern generative video models."
)


def _prune_upload_jobs(now: float | None = None) -> None:
    if now is None:
        now = time.time()
    to_delete = []
    for job_id, job in upload_jobs.items():
        status = job.get("status")
        finished_at = job.get("finished_at")
        if status in {"done", "failed"} and finished_at:
            if now - float(finished_at) > UPLOAD_JOB_RETENTION_SECONDS:
                to_delete.append(job_id)
    for job_id in to_delete:
        upload_jobs.pop(job_id, None)

    # Hard cap to avoid unbounded memory growth from queued/running metadata.
    if len(upload_jobs) > MAX_UPLOAD_JOBS:
        oldest = sorted(
            upload_jobs.items(),
            key=lambda item: float(item[1].get("created_at", now)),
        )
        overflow = len(upload_jobs) - MAX_UPLOAD_JOBS
        for job_id, _ in oldest[:overflow]:
            upload_jobs.pop(job_id, None)


@bp.get("/")
def index():
    return render_template("index.html", disclaimer=DISCLAIMER)


@bp.get("/about")
def about():
    return render_template("about.html", disclaimer=DISCLAIMER)


@bp.get("/api/score")
def api_score():
    return jsonify(state.score_dict())


@bp.get("/api/config")
def api_config():
    return jsonify(
        {
            "inference_stride": state.inference_stride,
            "upload_workers": UPLOAD_WORKERS,
            "upload_job_retention_seconds": UPLOAD_JOB_RETENTION_SECONDS,
            "max_upload_jobs": MAX_UPLOAD_JOBS,
            "frame_max_dim": FRAME_MAX_DIM,
            "mode": state.analyzer.mode,
        }
    )


@bp.get("/health")
def health():
    return jsonify({"ok": True, "service": "deepfake-detector"})


@bp.post("/analyze/frame")
def analyze_frame():
    start = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    image_base64 = payload.get("image_base64")
    if not image_base64:
        return jsonify({"ok": False, "error": "image_base64 is required"}), 400

    # Support raw base64 and data URLs like "data:image/jpeg;base64,..."
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError):
        return jsonify({"ok": False, "error": "Invalid base64 image"}), 400

    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"ok": False, "error": "Could not decode image"}), 400
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest > FRAME_MAX_DIM:
        scale = FRAME_MAX_DIM / float(longest)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    result = state.analyze_and_store(frame, smooth=True)
    response = {
        "ok": True,
        "fake_probability": result.fake_probability,
        "label": result.label,
        "face_detected": result.face_detected,
        "mode": result.mode,
        "timestamp": time.time(),
        "processing_ms": round((time.perf_counter() - start) * 1000, 2),
    }
    return jsonify(response)


@bp.post("/camera/start")
def camera_start():
    index = int(request.form.get("camera_index", 0))
    state.start_camera(index)
    return jsonify({"ok": True, "message": "Camera started"})


@bp.post("/camera/stop")
def camera_stop():
    state.stop_camera()
    return jsonify({"ok": True, "message": "Camera stopped"})


@bp.get("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = state.read_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            result = state.analyze_and_store(frame)
            overlay = state.analyzer.draw_overlay(frame, result)
            ok, buffer = cv2.imencode(".jpg", overlay)
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.post("/analyze/upload")
def analyze_upload():
    file = request.files.get("video")
    if file is None or file.filename == "":
        return jsonify({"ok": False, "error": "No video uploaded"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return jsonify({"ok": False, "error": "Unsupported video format"}), 400

    save_path = UPLOAD_DIR / f"upload_{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
    file.save(save_path)

    sample_stride = max(1, int(request.form.get("sample_stride", 5)))
    job_id = uuid.uuid4().hex
    with upload_jobs_lock:
        _prune_upload_jobs()
        upload_jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "created_at": time.time(),
            "filename": file.filename,
            "sample_stride": sample_stride,
        }

    def run_job() -> None:
        with upload_jobs_lock:
            if job_id in upload_jobs:
                upload_jobs[job_id]["status"] = "running"
                upload_jobs[job_id]["started_at"] = time.time()
        try:
            summary = state.analyze_video_file(
                video_path=save_path,
                original_filename=file.filename,
                sample_stride=sample_stride,
            )
            summary["disclaimer"] = DISCLAIMER
            with upload_jobs_lock:
                if job_id in upload_jobs:
                    upload_jobs[job_id]["status"] = "done"
                    upload_jobs[job_id]["finished_at"] = time.time()
                    upload_jobs[job_id]["result"] = summary
        except Exception as exc:
            with upload_jobs_lock:
                if job_id in upload_jobs:
                    upload_jobs[job_id]["status"] = "failed"
                    upload_jobs[job_id]["finished_at"] = time.time()
                    upload_jobs[job_id]["error"] = str(exc)
        finally:
            try:
                save_path.unlink(missing_ok=True)
            except OSError:
                pass

    upload_executor.submit(run_job)
    return jsonify({"ok": True, "job_id": job_id, "status": "queued"}), 202


@bp.get("/analyze/upload/<job_id>")
def analyze_upload_status(job_id: str):
    with upload_jobs_lock:
        _prune_upload_jobs()
        job = upload_jobs.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Job not found"}), 404
    return jsonify(job)


@bp.get("/analyze/upload/jobs")
def analyze_upload_jobs():
    include_results = request.args.get("include_results", "0") == "1"
    with upload_jobs_lock:
        _prune_upload_jobs()
        jobs = sorted(
            upload_jobs.values(),
            key=lambda item: float(item.get("created_at", 0.0)),
            reverse=True,
        )
        if not include_results:
            jobs = [
                {k: v for k, v in job.items() if k != "result"}
                for job in jobs
            ]
    return jsonify({"ok": True, "count": len(jobs), "jobs": jobs})


@bp.delete("/analyze/upload/<job_id>")
def analyze_upload_delete(job_id: str):
    with upload_jobs_lock:
        removed = upload_jobs.pop(job_id, None)
    if removed is None:
        return jsonify({"ok": False, "error": "Job not found"}), 404
    return jsonify({"ok": True, "deleted": job_id})
