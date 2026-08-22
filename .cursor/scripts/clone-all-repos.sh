#!/usr/bin/env bash
# Clone sibling repos into /workspace/projects for multi-repo cloud agent access.
set -euo pipefail

ROOT="/workspace/projects"
mkdir -p "$ROOT"

clone_if_missing() {
  local repo="$1"
  local name="${repo##*/}"
  if [ -d "$ROOT/$name/.git" ]; then
    echo "Already cloned: $name"
    return 0
  fi
  echo "Cloning $repo ..."
  git clone --depth 1 "https://github.com/$repo.git" "$ROOT/$name" || echo "WARN: failed to clone $repo"
}

# Primary repo is already checked out at /workspace.
for dep in \
  KRYPTON0078/KRYPTON0078 \
  KRYPTON0078/abva \
  KRYPTON0078/Agri-Link \
  KRYPTON0078/cofreseguro \
  KRYPTON0078/hes-agent-platform \
  KRYPTON0078/Jarvis \
  KRYPTON0078/Jarvis-AI \
  KRYPTON0078/Off-grid-Communication-System \
  KRYPTON0078/OPTIMIZED_PLFM_RADAR \
  KRYPTON0078/Physical-Smoke-Detector \
  KRYPTON0078/Python-Cloud-Keylogger-Educational- \
  KRYPTON0078/residential-esms-detectability \
  KRYPTON0078/ros-cyber \
  KRYPTON0078/RuView-Optimized \
  KRYPTON0078/SmartHome-CTF \
  KRYPTON0078/Smoking_Detection-Machine-Learning-Project
do
  clone_if_missing "$dep"
done

echo "Projects available under $ROOT"
