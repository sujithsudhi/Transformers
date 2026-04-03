# Transformers

Lightweight transformer training and inference workflows for:

- IMDB sentiment classification
- TinyStories decoder language modeling
- Checkpoint export and deployment prep for ONNX / TensorRT / Triton

## Project Layout

```text
Transformers/
|-- app/                  # Training entry-points
|-- configs/              # Shared and app-specific dataclass configs
|-- data/                 # IMDB and TinyStories data prep helpers
|-- docs/                 # Project notes and migration docs
|-- inference/            # Checkpoint export and inference scripts
|-- models/               # Model architectures used by the apps
|-- tool/deploy/          # Deployment helpers for ONNX / TensorRT / Triton
|-- training/             # Trainer integration, logging, plotting helpers
|-- val/                  # Validation entry-points
|-- viz/                  # Plot utilities for metrics and predictions
|-- results/              # Saved checkpoints, plots, and histories
|-- requirements.txt
`-- environment.yml
```

## Setup

This repo expects the sibling `transformer-core` and `trainer-core` packages beside it:

```text
Repos/
  Transformers/
  transformer-core/
  trainer-core/
```

Install with pip:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate transformers-model
```

Additional migration notes live in [docs/transformer-core-migration.md](/c:/Users/Sujith/Dev/Repos/Transformers/docs/transformer-core-migration.md).

## Common Commands

Train IMDB sentiment:

```bash
python -m app.train_sentiment
```

Train TinyStories decoder:

```bash
python -m app.train_decoder
```

Validate an IMDB checkpoint:

```bash
python -m val.validate \
  --config configs.imdb:IMDBConfig \
  --checkpoint results/imdb_transformer.pt \
  --split test \
  --output-dir results/validation
```

Run TinyStories inference / generation:

```bash
python -m inference.python.infer_tinystories \
  --checkpoint results/tiny_stories_transformer.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 128
```

Export a PyTorch checkpoint:

```bash
python -m inference.pytorch_checkpoint_exporter \
  --config configs.imdb:IMDBConfig \
  --checkpoint results/imdb_transformer.pt \
  --output inference/exports/imdb_checkpoint \
  --format npz
```

Deployment-specific helpers are documented further in [tool/deploy/README.md](/c:/Users/Sujith/Dev/Repos/Transformers/tool/deploy/README.md).

## Configuration

`configs/base.py` defines the shared config surface. App-specific configs in
`configs/imdb.py` and `configs/tinystories.py` inherit from it and keep the
project-wide structure consistent:

- `data`
- `model`
- `training`
- `dataloader`
- `optimizer`
- `loss`

The training scripts are config-driven. Update the active values in
`configs/imdb.py` or `configs/tinystories.py` and then run the matching entry
point.

## Outputs

Training runs can write:

- checkpoint files
- serialized config payloads inside checkpoints
- history JSON files
- loss / accuracy plots
- confusion matrices and class-distribution plots for IMDB

By default these artifacts land under `results/`, and those paths are controlled
from the config files.

## Experiment Tracking

Weights & Biases is optional. When installed, the training scripts respect:

- `WANDB_API_KEY`
- `WANDB_PROJECT`
- `WANDB_ENTITY`
- `WANDB_NAME`
- `WANDB_DISABLED`

You can also disable logging in the config or via `WANDB_DISABLED`.
