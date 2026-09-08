"""Smoke tests for Flask dashboard routes, including GET /health."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.server import create_app


def test_health_endpoint_reports_service_and_mode():
    client = create_app().test_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["service"] == "deepfake-detector"
    assert payload["mode"] in {"heuristic", "cnn"}
    assert payload["face_detector_backend"] in {"yunet", "haar"}
    assert "thresholds" in payload
    assert "fake_threshold" in payload["thresholds"]
    assert "checkpoint_present" in payload
    assert payload["camera_open"] is False


def test_index_and_about_render():
    client = create_app().test_client()
    index = client.get("/")
    about = client.get("/about")
    assert index.status_code == 200
    assert about.status_code == 200
    assert b"Deepfake Artifact Detector" in index.data
    assert b"Scope" in about.data


def test_api_config_exposes_thresholds():
    client = create_app().test_client()
    response = client.get("/api/config")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["mode"] in {"heuristic", "cnn"}
    assert 1 <= int(payload["inference_stride"]) <= 10
    assert "thresholds" in payload
