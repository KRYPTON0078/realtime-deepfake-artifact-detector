"""Dataset utilities for face-swap artifact classification."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class FaceArtifactDataset(Dataset):
    """Folder layout: root/real/*.jpg and root/fake_face_swap/*.jpg"""

    def __init__(self, root: Path, transform=TRAIN_TRANSFORM) -> None:
        self.root = Path(root)
        self.transform = transform
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
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(train_dir: Path, val_dir: Path, batch_size: int = 16):
    train_ds = FaceArtifactDataset(train_dir, transform=TRAIN_TRANSFORM)
    val_ds = FaceArtifactDataset(val_dir, transform=VAL_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
