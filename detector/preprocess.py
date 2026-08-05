"""Image preprocessing for CNN training and inference."""

from __future__ import annotations

import cv2
import numpy as np
import torch

INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_square(image: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def normalize_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """Convert HxWxC uint8 RGB image to CHW float32 normalized array."""
    arr = image_rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))


def preprocess_face(image_bgr: np.ndarray, size: int = INPUT_SIZE) -> torch.Tensor:
    """Convert a BGR face crop into a normalized model tensor without PIL."""
    rgb = bgr_to_rgb(image_bgr)
    resized = resize_square(rgb, size=size)
    chw = normalize_rgb(resized)
    tensor = torch.from_numpy(chw).unsqueeze(0)
    return tensor


def preprocess_rgb_image(image_rgb: np.ndarray, size: int = INPUT_SIZE) -> torch.Tensor:
    """Shared path for RGB images used by training datasets."""
    resized = resize_square(image_rgb, size=size)
    chw = normalize_rgb(resized)
    return torch.from_numpy(chw)
