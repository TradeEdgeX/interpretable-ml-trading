#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="hansenlovefiona017/ml_trading_bot_devcontainer:latest"

echo "🛠  Building devcontainer image: ${IMAGE_NAME}"
docker build \
  -f "${PROJECT_ROOT}/.devcontainer/Dockerfile" \
  -t "${IMAGE_NAME}" \
  "${PROJECT_ROOT}"

echo "✅ Build completed: ${IMAGE_NAME}"

