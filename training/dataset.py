"""Dataset utilities for face-swap artifact classification."""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from detector.preprocess import INPUT_SIZE, preprocess_rgb_image


def _jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def augment_pil(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.7:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.6)))
    if rng.random() < 0.5:
        image = _jpeg_compress(image, quality=rng.randint(35, 90))
    if rng.random() < 0.3:
        arr = np.array(image).astype(np.int16)
        noise = np.random.default_rng(rng.randint(0, 10_000_000)).integers(-18, 19, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
    # mild scale jitter via resize then center-ish crop back
    if rng.random() < 0.4:
        scale = rng.uniform(0.85, 1.15)
        width, height = image.size
        image = image.resize((max(8, int(width * scale)), max(8, int(height * scale))), Image.BILINEAR)
        image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    else:
        image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    return image


class FaceArtifactDataset(Dataset):
    """Folder layout: root/real/*.jpg and root/fake_face_swap/*.jpg"""

    def __init__(self, root: Path, train: bool = True, seed: int = 42) -> None:
        self.root = Path(root)
        self.train = train
        self.rng = random.Random(seed)
        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx = {"real": 0, "fake_face_swap": 1}

        for class_name, label in self.class_to_idx.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for path in class_dir.glob("*"):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.train:
            # per-item RNG derived from base seed + index for reproducibility in epoch order
            local_rng = random.Random(self.rng.randint(0, 10_000_000) + index)
            image = augment_pil(image, local_rng)
            rgb = np.array(image)
        else:
            rgb = np.array(image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))
        tensor = preprocess_rgb_image(rgb)
        return tensor, label

    def label_list(self) -> list[int]:
        return [label for _, label in self.samples]


def create_dataloaders(train_dir: Path, val_dir: Path, batch_size: int = 16, seed: int = 42):
    train_ds = FaceArtifactDataset(train_dir, train=True, seed=seed)
    val_ds = FaceArtifactDataset(val_dir, train=False, seed=seed)

    sampler = None
    shuffle = True
    if len(train_ds) > 0:
        labels = np.asarray(train_ds.label_list())
        class_count = np.bincount(labels, minlength=2).astype(np.float64)
        class_count[class_count == 0] = 1.0
        weights = 1.0 / class_count[labels]
        sampler = WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
