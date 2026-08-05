"""Calibrate fake/warn thresholds from a validation metrics JSON or live scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detector.model import build_model
from training.dataset import create_dataloaders
from training.metrics import best_threshold_by_f1, compute_binary_metrics


@torch.no_grad()
def collect_scores(model, loader, device):
    model.eval()
    labels = []
    scores = []
    for images, batch_labels in loader:
        images = images.to(device)
        probs = F.softmax(model(images), dim=1)[:, 1]
        labels.extend(batch_labels.tolist())
        scores.extend(probs.detach().cpu().tolist())
    return np.asarray(labels), np.asarray(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate detector thresholds on validation data")
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "config.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "artifact_detector.pt")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "calibrated_thresholds.json",
    )
    parser.add_argument("--warn-gap", type=float, default=0.15, help="warn = fake - warn_gap")
    parser.add_argument("--hysteresis-gap", type=float, default=0.15, help="exit_fake = fake - gap")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["architecture"], config["num_classes"]).to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, val_loader = create_dataloaders(
        ROOT / config["train_dir"],
        ROOT / config["val_dir"],
        config["batch_size"],
    )
    if len(val_loader.dataset) == 0:
        raise SystemExit("No validation images available for calibration")

    y_true, scores = collect_scores(model, val_loader, device)
    fake_threshold, best = best_threshold_by_f1(y_true, scores)
    warn_threshold = max(0.05, fake_threshold - args.warn_gap)
    exit_fake = max(0.05, fake_threshold - args.hysteresis_gap)
    enter_fake = min(0.99, fake_threshold + 0.05)

    payload = {
        "fake_threshold": float(fake_threshold),
        "warn_threshold": float(warn_threshold),
        "enter_fake": float(enter_fake),
        "exit_fake": float(exit_fake),
        "metrics_at_fake_threshold": best.to_dict(),
        "metrics_at_default_0_55": compute_binary_metrics(y_true, scores, 0.55).to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
