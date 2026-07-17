"""Flask routes for the deepfake artifact detector demo."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
from flask import Blueprint, Response, jsonify, render_template, request

from app.state import UPLOAD_DIR, state

bp = Blueprint("main", __name__)

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

    save_path = UPLOAD_DIR / f"upload_{int(time.time())}{suffix}"
    file.save(save_path)

    cap = cv2.VideoCapture(str(save_path))
    if not cap.isOpened():
        return jsonify({"ok": False, "error": "Could not open uploaded video"}), 400

    frame_scores = []
    frame_index = 0
    state.analyzer.smoother.reset()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % 5 != 0:
            frame_index += 1
            continue
        result = state.analyze_and_store(frame, smooth=False)
        frame_scores.append(
            {
                "frame": frame_index,
                "fake_probability": result.fake_probability,
                "label": result.label,
                "face_detected": result.face_detected,
            }
        )
        frame_index += 1
    cap.release()

    if not frame_scores:
        summary = {"ok": False, "error": "No frames processed"}
    else:
        avg_score = sum(item["fake_probability"] for item in frame_scores) / len(frame_scores)
        manipulated = sum(1 for item in frame_scores if item["label"] == "likely_manipulated")
        summary = {
            "ok": True,
            "filename": file.filename,
            "frames_analyzed": len(frame_scores),
            "average_fake_probability": round(avg_score, 3),
            "manipulated_frame_ratio": round(manipulated / len(frame_scores), 3),
            "mode": state.analyzer.mode,
            "disclaimer": DISCLAIMER,
            "samples": frame_scores[:20],
        }
    state.upload_summary = summary
    return jsonify(summary)
