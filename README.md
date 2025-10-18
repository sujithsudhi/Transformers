# Foundation-Model

Sentiment-classification sandbox built on top of a lightweight transformer architecture. The project fine-tunes a generic foundation model on the IMDB reviews dataset using a configurable PyTorch training loop.

## Project layout

```
Foundation-Model/
├── data/                # Dataset loaders and download helpers
├── models/              # Transformer model + training utilities
├── notebooks/           # Jupyter exploration (optional)
├── results/             # Persisted metrics and plots
├── requirements.txt     # Python dependencies
├── train_sentiment.py   # IMDB training entry-point
└── README.md
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training

```bash
python train_sentiment.py --config configs.imdb:IMDBConfig
```

Override any parameter directly:

```bash
python train_sentiment.py \
  --config configs.imdb:IMDBConfig \
  --epochs 3 \
  --batch-size 16 \
  --history-out results/custom_history.json
```

The IMDB config inherits shared defaults defined in `configs/base.py`. For new applications, create a module (see `configs/foundation.py`) that subclasses the base dataclasses, point `train_sentiment.py` at it via `--config`, and adjust dataset/model hyperparameters as needed. Loss curves and metrics default to the paths declared on the config (`history_path`, `plot_path`).
