#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model.onnx> <output.plan> [additional trtexec args...]"
  exit 1
fi

MODEL_PATH="$1"
ENGINE_PATH="$2"
shift 2

TRTEXEC_BIN="${TRTEXEC_BIN:-trtexec}"

echo "Building TensorRT engine:"
echo "  ONNX model : ${MODEL_PATH}"
echo "  Output plan: ${ENGINE_PATH}"

"${TRTEXEC_BIN}" \
  --onnx="${MODEL_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --workspace=4096 \
  --explicitBatch \
  "$@"

echo "TensorRT engine written to ${ENGINE_PATH}"
