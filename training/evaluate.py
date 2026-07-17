"""Evaluate a trained artifact detector checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.model import build_model
from training.dataset import create_dataloaders
from training.train import evaluate


def main(config_path: Path, model_path: Path) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["architecture"], config["num_classes"]).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, val_loader = create_dataloaders(
        ROOT / config["train_dir"],
        ROOT / config["val_dir"],
        config["batch_size"],
    )
    val_acc = evaluate(model, val_loader, device)
    print(f"Validation accuracy: {val_acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "config.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "artifact_detector.pt")
    args = parser.parse_args()
    main(args.config, args.model)
