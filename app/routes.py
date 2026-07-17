"""Flask routes for the deepfake artifact detector demo."""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from flask import Blueprint, Response, jsonify, render_template, request

from app.state import UPLOAD_DIR, state

bp = Blueprint("main", __name__)
UPLOAD_WORKERS = max(1, int(os.getenv("UPLOAD_ANALYSIS_WORKERS", "2")))
upload_executor = ThreadPoolExecutor(max_workers=UPLOAD_WORKERS)
upload_jobs: dict[str, dict] = {}
upload_jobs_lock = threading.Lock()

DISCLAIMER = (
    "Detector trained for face-swap artifact patterns; "
    "not validated against all modern generative video models."
)


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
            "mode": state.analyzer.mode,
        }
    )


@bp.get("/health")
def health():
    return jsonify({"ok": True, "service": "deepfake-detector"})


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
        job = upload_jobs.get(job_id)
    if job is None:
        return jsonify({"ok": False, "error": "Job not found"}), 404
    return jsonify(job)
