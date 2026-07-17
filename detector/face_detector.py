"""Face detection utilities using OpenCV Haar cascades."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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
    """Detect frontal faces in BGR frames."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(60, 60),
        )
        detections = [
            FaceDetection(int(x), int(y), int(w), int(h))
            for x, y, w, h in faces
        ]
        if not detections:
            return []
        detections.sort(key=lambda item: item.w * item.h, reverse=True)
        return detections

    def crop_largest_face(self, frame: np.ndarray) -> tuple[np.ndarray | None, FaceDetection | None]:
        detections = self.detect(frame)
        if not detections:
            return None, None
        face = detections[0]
        x, y, w, h = face.box
        crop = frame[y : y + h, x : x + w]
        return crop, face
