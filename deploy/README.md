# Deployment Utilities

This directory gathers scripts and helpers for exporting and serving the fine-tuned
`TransformersModel`.

## Export Workflows

- `onnx_export.py` — exports a PyTorch checkpoint to ONNX. Optionally produces a TFLite
  artifact when the required TensorFlow tooling is installed.
- `infer_onnx_ort_trt.py` — runs inference on an ONNX graph using ONNX Runtime with
  TensorRT, CUDA, and CPU execution providers.
- `build_trt_engine.sh` — wraps `trtexec` to build a TensorRT engine from the ONNX graph.
- `infer_trt_engine.py` — loads a TensorRT engine and performs inference via the TensorRT
  Python API.

## Triton Inference Server

- `triton/model_repository/bert_sst2/1` — place the exported ONNX model at
  `model.onnx`. The accompanying `config.pbtxt` defines the input and output tensor
  shapes expected by Triton.
- `triton/run_triton_jetson.sh` — add orchestration scripts (e.g., Docker launchers) here
  when serving from Jetson-class devices.

## Tokenizer Assets

The `tokenizer/` directory is reserved for serialized tokenizer files that must accompany
the exported model. Add tokenizer JSON/vocabulary files after export.
