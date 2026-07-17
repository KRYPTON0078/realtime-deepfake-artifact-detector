"""Shared application state for Flask routes."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from detector.inference import AnalysisResult, DeepfakeAnalyzer

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


class AppState:
    def __init__(self) -> None:
        self.analyzer = DeepfakeAnalyzer()
        self.camera = None
        self.camera_lock = threading.Lock()
        self.latest_score = ScoreSnapshot(0.0, "idle", False, self.analyzer.mode, time.time())
        self.upload_summary: dict | None = None
        self.inference_stride = 2
        self._frame_counter = 0
        self._last_result: AnalysisResult | None = None

    def start_camera(self, camera_index: int = 0) -> None:
        with self.camera_lock:
            self.stop_camera()
            self.camera = cv2.VideoCapture(camera_index)
            self.analyzer.smoother.reset()

    def stop_camera(self) -> None:
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def read_frame(self):
        with self.camera_lock:
            if self.camera is None or not self.camera.isOpened():
                return None
            ok, frame = self.camera.read()
            if not ok:
                return None
            return frame

    def analyze_and_store(self, frame, smooth: bool = True) -> AnalysisResult:
        self._frame_counter += 1
        if self._frame_counter % self.inference_stride != 0 and self._last_result is not None:
            return self._last_result
        result = self.analyzer.analyze_frame(frame, smooth=smooth)
        self._last_result = result
        self.latest_score = ScoreSnapshot(
            fake_probability=result.fake_probability,
            label=result.label,
            face_detected=result.face_detected,
            mode=result.mode,
            timestamp=time.time(),
        )
        return result

    def score_dict(self) -> dict:
        return asdict(self.latest_score)


state = AppState()
