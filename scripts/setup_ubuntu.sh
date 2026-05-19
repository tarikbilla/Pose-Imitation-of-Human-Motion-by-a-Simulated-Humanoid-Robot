#!/usr/bin/env bash
# One-shot setup script for Ubuntu 22.04 / 24.04.
# Usage:  bash scripts/setup_ubuntu.sh
set -euo pipefail

PY=${PY:-python3.11}

echo "==> Checking system packages..."
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Installing Python 3.11 + system libs (requires sudo)..."
  sudo apt update
  sudo apt install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git ffmpeg libgl1 libglib2.0-0 v4l-utils
fi

echo "==> Creating virtualenv (.venv)..."
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip + installing dependencies..."
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

echo "==> Detecting cameras..."
if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl --list-devices || true
else
  ls -1 /dev/video* 2>/dev/null || echo "No /dev/video* devices found."
fi

echo ""
echo "✔ Setup complete."
echo "   Activate environment:  source .venv/bin/activate"
echo "   Run demo:              python run.py --no-webots"
echo "   Run full pipeline:     python run.py"
