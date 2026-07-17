"""Image preprocessing for CNN inference."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torchvision import transforms


INPUT_SIZE = 224

INFER_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def preprocess_face(image_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR face crop into a normalized model tensor."""
    rgb = bgr_to_rgb(image_bgr)
    tensor = INFER_TRANSFORM(rgb)
    return tensor.unsqueeze(0)
