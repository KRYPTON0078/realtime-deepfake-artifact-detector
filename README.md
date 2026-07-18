# Real-Time Deepfake Artifact Detector

Real-time vision pipeline for detecting visual artifacts associated with deepfake / synthetic media manipulation.

## Problem

Synthetic media can look convincing at a glance. A practical detector should inspect frames in near real time for artifact cues that indicate generation or manipulation.

## Approach

- Frame-level or short-window analysis of visual artifact signals
- Machine learning / computer vision classification path for suspicious frames
- Designed for interactive / streaming use rather than offline-only batch scoring

## Tech stack

- Python
- Computer vision libraries (OpenCV family)
- Deep learning framework (TensorFlow and/or PyTorch, depending on model path in repo)

## Status

Active development repository used to practice real-time media forensics and edge-deployable vision inference patterns.

## My role

Design and implementation of the detection pipeline, experimentation with artifact features, and iteration toward low-latency inference suitable for interactive demos.

## Relevance

Supports **Edge AI** and trustworthy vision themes: on-stream detection, media integrity cues, and practical ML deployment constraints.
