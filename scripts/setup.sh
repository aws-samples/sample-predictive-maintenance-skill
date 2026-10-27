#!/usr/bin/env bash
# Setup script for PdM model project.
# Copies bundled assets into the current working directory and installs dependencies.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Copying project files..."
# pyproject.toml + uv.lock + .python-version pin the exact training environment
# (Python and AutoGluon versions) that the inference container and edge
# component reproduce. Do not re-resolve them independently.
cp "$SKILL_DIR"/pyproject.toml "$SKILL_DIR"/uv.lock "$SKILL_DIR"/.python-version .
cp "$SKILL_DIR"/scripts/save_model.sh .
cp -r "$SKILL_DIR/pdm" ./pdm

echo "Creating directory structure..."
mkdir -p data anomaly_detection fault_prediction

echo "Creating virtual environment and installing dependencies..."
uv sync

echo "Setup complete."
