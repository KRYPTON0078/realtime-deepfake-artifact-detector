"""Train the face-swap artifact detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.model import build_model
from training.dataset import create_dataloaders


def train(config_path: Path) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    train_dir = ROOT / config["train_dir"]
    val_dir = ROOT / config["val_dir"]
    output_model = ROOT / config["output_model"]
    output_model.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["architecture"], config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])

    train_loader, val_loader = create_dataloaders(train_dir, val_dir, config["batch_size"])
    if len(train_loader.dataset) == 0:
        raise RuntimeError(
            f"No training images found in {train_dir}. "
            "Run scripts/generate_demo_dataset.py or add real/fake_face_swap folders."
        )

    best_val_acc = 0.0
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / max(total, 1)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{config['epochs']} loss={running_loss / max(total, 1):.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": config["architecture"],
                    "class_names": config["class_names"],
                    "val_acc": val_acc,
                },
                output_model,
            )
            print(f"Saved best model to {output_model}")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    if len(loader.dataset) == 0:
        return 0.0
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / max(total, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "training" / "config.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()
    train(args.config)
