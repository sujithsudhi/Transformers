#!/usr/bin/env bash

# Placeholder launcher for Triton Inference Server on Jetson devices.
# Set TRITON_IMAGE to your chosen Triton container (e.g. nvcr.io/nvidia/tritonserver:xx.xx-py3).
# Example usage:
#   TRITON_IMAGE=nvcr.io/nvidia/tritonserver:23.12-py3 \
#   ./run_triton_jetson.sh

set -euo pipefail

if [[ -z "${TRITON_IMAGE:-}" ]]; then
  echo "Set TRITON_IMAGE to the desired Triton container image."
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_REPO="${REPO_ROOT}/tool/deploy/triton/model_repository"

docker run --rm -it \
  --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "${MODEL_REPO}:/models" \
  "${TRITON_IMAGE}" \
  tritonserver --model-repository=/models
