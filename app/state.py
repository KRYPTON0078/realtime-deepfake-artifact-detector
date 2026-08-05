"""Shared application state for Flask routes."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from detector.inference import AnalysisResult, DeepfakeAnalyzer
from detector.temporal import TemporalSmoother

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@dataclass
class ScoreSnapshot:
    fake_probability: float
    label: str
    face_detected: bool
    mode: str
    timestamp: float
    face_box: tuple[int, int, int, int] | None = None
    face_confidence: float | None = None
    path: str = "idle"


class AppState:
    def __init__(self) -> None:
        self.analyzer = DeepfakeAnalyzer()
        self.camera = None
        self.camera_lock = threading.Lock()
        self.latest_score = ScoreSnapshot(0.0, "idle", False, self.analyzer.mode, time.time())
        self.upload_summary: dict | None = None
        self.inference_stride = max(1, int(os.getenv("INFERENCE_STRIDE", "2")))
        self._frame_counter = 0
        self._last_result: AnalysisResult | None = None
        self._smoothers = {
            "webcam": TemporalSmoother(
                enter_fake=self.analyzer.thresholds["enter_fake"],
                exit_fake=self.analyzer.thresholds["exit_fake"],
                warn_threshold=self.analyzer.thresholds["warn_threshold"],
            ),
            "mobile": TemporalSmoother(
                enter_fake=self.analyzer.thresholds["enter_fake"],
                exit_fake=self.analyzer.thresholds["exit_fake"],
                warn_threshold=self.analyzer.thresholds["warn_threshold"],
            ),
            "upload": TemporalSmoother(
                enter_fake=self.analyzer.thresholds["enter_fake"],
                exit_fake=self.analyzer.thresholds["exit_fake"],
                warn_threshold=self.analyzer.thresholds["warn_threshold"],
            ),
        }

        # Latest-frame stream worker state
        self._stream_thread: threading.Thread | None = None
        self._stream_stop = threading.Event()
        self._latest_overlay_jpeg: bytes | None = None
        self._overlay_lock = threading.Lock()
        self._target_fps = max(1.0, float(os.getenv("STREAM_TARGET_FPS", "12")))

    def start_camera(self, camera_index: int = 0) -> dict:
        with self.camera_lock:
            self.stop_camera(join_worker=True)
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                camera.release()
                return {"ok": False, "error": f"Could not open camera index {camera_index}"}
            self.camera = camera
            self._smoothers["webcam"].reset()
            self.analyzer.smoother.reset()
            self._stream_stop.clear()
            self._stream_thread = threading.Thread(
                target=self._stream_worker_loop,
                name="webcam-stream-worker",
                daemon=True,
            )
            self._stream_thread.start()
            return {"ok": True, "message": "Camera started", "camera_index": camera_index}

    def stop_camera(self, join_worker: bool = False) -> None:
        self._stream_stop.set()
        if join_worker and self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2.0)
        self._stream_thread = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        with self._overlay_lock:
            self._latest_overlay_jpeg = None

    def read_frame(self):
        with self.camera_lock:
            if self.camera is None or not self.camera.isOpened():
                return None
            ok, frame = self.camera.read()
            if not ok:
                return None
            return frame

    def _stream_worker_loop(self) -> None:
        min_interval = 1.0 / self._target_fps
        while not self._stream_stop.is_set():
            started = time.perf_counter()
            frame = self.read_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            result = self.analyze_and_store(frame, smooth=True, path="webcam")
            overlay = self.analyzer.draw_overlay(frame, result)
            ok, buffer = cv2.imencode(".jpg", overlay)
            if ok:
                with self._overlay_lock:
                    self._latest_overlay_jpeg = buffer.tobytes()
            elapsed = time.perf_counter() - started
            time.sleep(max(0.0, min_interval - elapsed))

    def get_latest_overlay_jpeg(self) -> bytes | None:
        with self._overlay_lock:
            return self._latest_overlay_jpeg

    def analyze_and_store(
        self,
        frame,
        smooth: bool = True,
        path: str = "webcam",
    ) -> AnalysisResult:
        self._frame_counter += 1
        if (
            path == "webcam"
            and self._frame_counter % self.inference_stride != 0
            and self._last_result is not None
        ):
            return self._last_result

        # Temporarily swap smoother for path isolation while reusing one model.
        original = self.analyzer.smoother
        self.analyzer.smoother = self._smoothers.get(path, original)
        try:
            result = self.analyzer.analyze_frame(frame, smooth=smooth)
        finally:
            self.analyzer.smoother = original

        if path == "webcam":
            self._last_result = result
        self.latest_score = ScoreSnapshot(
            fake_probability=result.fake_probability,
            label=result.label,
            face_detected=result.face_detected,
            mode=result.mode,
            timestamp=time.time(),
            face_box=result.face_box,
            face_confidence=result.face_confidence,
            path=path,
        )
        return result

    def score_dict(self) -> dict:
        return asdict(self.latest_score)

    def health_dict(self) -> dict:
        camera_open = False
        with self.camera_lock:
            camera_open = self.camera is not None and self.camera.isOpened()
        return {
            "ok": True,
            "service": "deepfake-detector",
            "mode": self.analyzer.mode,
            "checkpoint_present": self.analyzer.model_path.exists(),
            "face_detector_backend": self.analyzer.face_detector.backend,
            "camera_open": camera_open,
            "stream_worker_alive": bool(self._stream_thread and self._stream_thread.is_alive()),
            "thresholds": self.analyzer.thresholds,
        }

    def analyze_video_file(
        self,
        video_path: Path,
        original_filename: str,
        sample_stride: int = 5,
        max_frames: int = 400,
    ) -> dict:
        """Analyze sampled frames from a saved video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"ok": False, "error": "Could not open uploaded video"}

        frame_scores = []
        frame_index = 0
        analyzed = 0
        self._smoothers["upload"].reset()
        while analyzed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % sample_stride != 0:
                frame_index += 1
                continue
            result = self.analyze_and_store(frame, smooth=False, path="upload")
            frame_scores.append(
                {
                    "frame": frame_index,
                    "fake_probability": result.fake_probability,
                    "label": result.label,
                    "face_detected": result.face_detected,
                }
            )
            analyzed += 1
            frame_index += 1
        cap.release()

        if not frame_scores:
            return {"ok": False, "error": "No frames processed"}

        avg_score = sum(item["fake_probability"] for item in frame_scores) / len(frame_scores)
        manipulated = sum(1 for item in frame_scores if item["label"] == "likely_manipulated")
        summary = {
            "ok": True,
            "filename": original_filename,
            "frames_analyzed": len(frame_scores),
            "average_fake_probability": round(avg_score, 3),
            "manipulated_frame_ratio": round(manipulated / len(frame_scores), 3),
            "mode": self.analyzer.mode,
            "samples": frame_scores[:20],
        }
        self.upload_summary = summary
        return summary


state = AppState()
