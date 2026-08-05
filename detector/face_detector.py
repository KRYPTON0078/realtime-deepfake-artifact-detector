"""Face detection with OpenCV YuNet and Haar fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from detector.face_crop import CropConfig, crop_with_margin

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YUNET_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"


@dataclass
class FaceDetection:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class FaceDetector:
    """Detect faces in BGR frames using YuNet when available, else Haar."""

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        min_confidence: float = 0.55,
        yunet_path: Path | None = None,
        crop_config: CropConfig | None = None,
    ) -> None:
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.min_confidence = min_confidence
        self.crop_config = crop_config or CropConfig()
        self.backend = "haar"
        self._yunet = None
        self._input_size = (320, 320)

        model_path = yunet_path or DEFAULT_YUNET_PATH
        if model_path.exists() and hasattr(cv2, "FaceDetectorYN"):
            detector = cv2.FaceDetectorYN.create(
                str(model_path),
                "",
                self._input_size,
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                top_k=5000,
            )
            self._yunet = detector
            self.backend = "yunet"

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def _detect_yunet(self, frame: np.ndarray) -> list[FaceDetection]:
        assert self._yunet is not None
        height, width = frame.shape[:2]
        self._yunet.setInputSize((width, height))
        _, faces = self._yunet.detect(frame)
        if faces is None:
            return []
        detections: list[FaceDetection] = []
        for face in faces:
            x, y, w, h = [int(v) for v in face[:4]]
            confidence = float(face[-1])
            if confidence < self.min_confidence:
                continue
            x = max(0, x)
            y = max(0, y)
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            detections.append(FaceDetection(x, y, w, h, confidence=confidence))
        detections.sort(key=lambda item: item.w * item.h, reverse=True)
        return detections

    def _detect_haar(self, frame: np.ndarray) -> list[FaceDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(60, 60),
        )
        detections = [
            FaceDetection(int(x), int(y), int(w), int(h), confidence=1.0)
            for x, y, w, h in faces
        ]
        detections.sort(key=lambda item: item.w * item.h, reverse=True)
        return detections

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        if self._yunet is not None:
            detections = self._detect_yunet(frame)
            if detections:
                return detections
        return self._detect_haar(frame)

    def crop_largest_face(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray | None, FaceDetection | None]:
        detections = self.detect(frame)
        if not detections:
            return None, None
        face = detections[0]
        if face.confidence < self.min_confidence and self.backend == "yunet":
            return None, None
        crop = crop_with_margin(frame, face, self.crop_config)
        if crop is None:
            return None, None
        return crop, face
