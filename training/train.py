"""Train the face-swap artifact detector."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.model import build_model
from training.dataset import create_dataloaders
from training.metrics import best_threshold_by_f1, compute_binary_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_param_groups(model: nn.Module, head_lr: float, backbone_lr: float):
    if hasattr(model, "classifier"):
        head_params = list(model.classifier.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]
    return [{"params": model.parameters(), "lr": head_lr}]


@torch.no_grad()
def collect_val_scores(model, loader, device):
    model.eval()
    labels = []
    scores = []
    for images, batch_labels in loader:
        images = images.to(device)
        probs = F.softmax(model(images), dim=1)[:, 1]
        labels.extend(batch_labels.tolist())
        scores.extend(probs.detach().cpu().tolist())
    return np.asarray(labels), np.asarray(scores)


def train(config_path: Path) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    train_dir = ROOT / config["train_dir"]
    val_dir = ROOT / config["val_dir"]
    output_model = ROOT / config["output_model"]
    output_model.parent.mkdir(parents=True, exist_ok=True)
    history_path = output_model.with_suffix(".history.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["architecture"], config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    head_lr = float(config["learning_rate"])
    backbone_lr = float(config.get("backbone_lr", head_lr * 0.1))
    optimizer = Adam(split_param_groups(model, head_lr, backbone_lr))

    train_loader, val_loader = create_dataloaders(
        train_dir,
        val_dir,
        config["batch_size"],
        seed=seed,
    )
    if len(train_loader.dataset) == 0:
        raise RuntimeError(
            f"No training images found in {train_dir}. "
            "Run scripts/generate_demo_dataset.py or add real/fake_face_swap folders."
        )

    best_val_f1 = -1.0
    patience = int(config.get("early_stop_patience", 5))
    stale_epochs = 0
    history = []
    freeze_epochs = int(config.get("freeze_backbone_epochs", 0))

    for epoch in range(config["epochs"]):
        if hasattr(model, "features") and freeze_epochs > 0:
            freeze = epoch < freeze_epochs
            for param in model.features.parameters():
                param.requires_grad = not freeze

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
        y_true, scores = collect_val_scores(model, val_loader, device)
        if len(y_true) == 0:
            val_metrics = None
            val_f1 = 0.0
            val_acc = 0.0
            best_threshold = 0.55
        else:
            best_threshold, best_metrics = best_threshold_by_f1(y_true, scores)
            val_metrics = best_metrics.to_dict()
            val_f1 = best_metrics.f1
            val_acc = best_metrics.accuracy

        epoch_record = {
            "epoch": epoch + 1,
            "loss": running_loss / max(total, 1),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "best_threshold": best_threshold,
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch + 1}/{config['epochs']} "
            f"loss={epoch_record['loss']:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f} val_f1={val_f1:.3f}"
        )

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            stale_epochs = 0
            warn_threshold = max(0.05, float(best_threshold) - 0.15)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": config["architecture"],
                    "class_names": config["class_names"],
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "fake_threshold": float(best_threshold),
                    "warn_threshold": warn_threshold,
                    "seed": seed,
                    "history": history,
                },
                output_model,
            )
            print(f"Saved best model to {output_model} (val_f1={val_f1:.3f})")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Wrote training history to {history_path}")


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
