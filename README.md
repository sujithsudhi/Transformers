# Transformers

Lightweight transformer fine-tuning stack for IMDB sentiment classification, plus
deployment utilities for exporting the trained model to ONNX/TensorRT/Triton.

## Project Layout

```
Transformers/
├── configs/             # Base + application configs (IMDB, Transformers scene demos)
├── data/                # IMDB dataset wrapper, download helpers
├── models/              # Transformer backbone + training utilities
├── scene_data/          # Scene metadata helpers (legacy research demos)
├── scripts/             # Extended training scripts (e.g., generic scene workflow)
├── tool/deploy/         # Export + inference helpers (ONNX, TensorRT, Triton)
├── viz/                 # Plotting utilities for training/eval metrics
├── results/             # Persisted metrics and plots
├── requirements.txt     # Python dependencies
├── train_sentiment.py   # IMDB training entry-point
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install extra packages if you plan to export/run ONNX or TensorRT models:

```bash
python -m pip install onnxruntime-gpu onnx onnx-tf tensorflow tensorrt pycuda
```

## Training (IMDB Sentiment)

```bash
python train_sentiment.py
```

`train_sentiment.py` loads `configs.imdb:IMDBConfig`, which captures dataset path,
tokenisation, model hyperparameters, and training knobs. `data/imdb.py` will download
the IMDB dataset (via Hugging Face `datasets`) into `data/imdb/` on first run. Cached
JSONL files remain ignored by Git.

Metrics and training curves are written to the paths defined by the active config
(`history_path`, `plot_path`).

## Generic Scene Workflow

`scripts/train_transformers.py` demonstrates a more generic configuration-driven
workflow: pass any `AppConfig` subclass (e.g., `configs.transformers:TransformersConfig`)
and the script builds dataloaders, optimizer, and loss from the config. Scene metadata
helpers live in `scene_data/`.

## Deployment

The `tool/deploy/` directory hosts export and inference scripts:

- `onnx_export.py` — load a PyTorch checkpoint and export to ONNX (optionally TFLite).
- `infer_onnx_ort_trt.py` — quick inference harness using ONNX Runtime with TensorRT/CUDA.
- `build_trt_engine.sh` / `infer_trt_engine.py` — build and run TensorRT engines via `trtexec`.
- `triton/` — Triton Inference Server scaffolding. Drop the exported ONNX model into
  `tool/deploy/triton/model_repository/bert_sst2/1/model.onnx` and launch Triton via
  `tool/deploy/triton/run_triton_jetson.sh`.
- `tokenizer/` — store tokenizer assets alongside exported models.

## Visualisation

Use helpers in `viz/` (e.g., `viz.plots.plot_loss_curves`) to quickly chart loss curves,
confusion matrices, or prediction distributions during evaluation notebooks or scripts.

## Customisation

- Modify base defaults in `configs/base.py` and subclass them per application.
- `scene_data/` components can be adapted for other structured datasets requiring the
  TransformersModel backbone.
- Update deployment scripts with project-specific namespaces, TensorRT profiles, or Triton
  configuration as you evolve the model.***
