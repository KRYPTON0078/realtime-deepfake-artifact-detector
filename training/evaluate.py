"""Evaluate a trained artifact detector checkpoint with rich metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
def collect_scores(model, loader, device) -> tuple[list[int], list[float]]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    for images, batch_labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)[:, 1]
        labels.extend(batch_labels.tolist())
        scores.extend(probs.detach().cpu().tolist())
    return labels, scores


def main(config_path: Path, model_path: Path, output_json: Path | None) -> None:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["architecture"], config["num_classes"]).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, val_loader = create_dataloaders(
        ROOT / config["train_dir"],
        ROOT / config["val_dir"],
        config["batch_size"],
    )
    if len(val_loader.dataset) == 0:
        raise SystemExit(f"No validation images found under {ROOT / config['val_dir']}")

    labels, scores = collect_scores(model, val_loader, device)
    import numpy as np

    y_true = np.asarray(labels)
    y_scores = np.asarray(scores)
    default_metrics = compute_binary_metrics(y_true, y_scores, threshold=0.55)
    best_threshold, best_metrics = best_threshold_by_f1(y_true, y_scores)

    report = {
        "model_path": str(model_path),
        "architecture": checkpoint.get("architecture", config["architecture"]),
        "checkpoint_val_acc": checkpoint.get("val_acc"),
        "checkpoint_val_f1": checkpoint.get("val_f1"),
        "default_threshold_metrics": default_metrics.to_dict(),
        "best_f1_threshold": best_threshold,
        "best_f1_metrics": best_metrics.to_dict(),
        "num_val_samples": int(len(y_true)),
    }
    print(json.dumps(report, indent=2))
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote metrics report to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "config.yaml")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "artifact_detector.pt")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "models" / "eval_metrics.json",
        help="Optional path to write the metrics JSON report",
    )
    args = parser.parse_args()
    main(args.config, args.model, args.output_json)
