"""Inference pipeline for real-time deepfake artifact scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from detector.face_detector import FaceDetection, FaceDetector
from detector.model import build_model
from detector.preprocess import preprocess_face
from detector.temporal import TemporalSmoother

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "artifact_detector.pt"
FAKE_THRESHOLD = 0.55
WARN_THRESHOLD = 0.40


@dataclass
class AnalysisResult:
    fake_probability: float
    label: str
    face_detected: bool
    mode: str
    face_box: tuple[int, int, int, int] | None = None


class DeepfakeAnalyzer:
    """Analyze frames for face-swap style manipulation artifacts."""

    def __init__(
        self,
        model_path: Path | None = None,
        architecture: str = "mobilenet_v2",
        device: str | None = None,
    ) -> None:
        self.face_detector = FaceDetector()
        self.smoother = TemporalSmoother(window_size=8)
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.mode = "heuristic"
        self._load_model(architecture)

    def _load_model(self, architecture: str) -> None:
        if not self.model_path.exists():
            return
        self.model = build_model(architecture=architecture, num_classes=2)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.mode = "cnn"

    @staticmethod
    def _heuristic_score(face_bgr: np.ndarray) -> float:
        """Fallback score based on spatial artifact cues when no trained model exists."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = 1.0 - min(laplacian_var / 500.0, 1.0)

        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        saturation_std = float(np.std(hsv[:, :, 1]) / 255.0)
        color_score = min(saturation_std * 2.0, 1.0)

        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.count_nonzero(edges) / edges.size)
        boundary_score = min(edge_density * 4.0, 1.0)

        score = 0.45 * blur_score + 0.30 * color_score + 0.25 * boundary_score
        return float(max(0.0, min(score, 1.0)))

    def _predict_cnn(self, face_bgr: np.ndarray) -> float:
        assert self.model is not None
        tensor = preprocess_face(face_bgr).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
        return float(probs[1].item())

    def analyze_frame(self, frame: np.ndarray, smooth: bool = True) -> AnalysisResult:
        face_crop, detection = self.face_detector.crop_largest_face(frame)
        if face_crop is None or detection is None:
            return AnalysisResult(
                fake_probability=0.0,
                label="no_face",
                face_detected=False,
                mode=self.mode,
                face_box=None,
            )

        if self.model is not None:
            raw_score = self._predict_cnn(face_crop)
        else:
            raw_score = self._heuristic_score(face_crop)

        score = self.smoother.update(raw_score) if smooth else raw_score
        label = self._label_from_score(score)
        return AnalysisResult(
            fake_probability=score,
            label=label,
            face_detected=True,
            mode=self.mode,
            face_box=detection.box,
        )

    @staticmethod
    def _label_from_score(score: float) -> str:
        if score >= FAKE_THRESHOLD:
            return "likely_manipulated"
        if score >= WARN_THRESHOLD:
            return "suspicious"
        return "likely_real"

    def draw_overlay(self, frame: np.ndarray, result: AnalysisResult) -> np.ndarray:
        output = frame.copy()
        if result.face_box:
            x, y, w, h = result.face_box
            color = self._color_for_label(result.label)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            text = f"{result.label} ({result.fake_probability:.2f}) [{result.mode}]"
            cv2.putText(
                output,
                text,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                output,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return output

    @staticmethod
    def _color_for_label(label: str) -> tuple[int, int, int]:
        if label == "likely_manipulated":
            return (0, 0, 255)
        if label == "suspicious":
            return (0, 165, 255)
        return (0, 200, 0)
