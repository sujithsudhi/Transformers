# Transformers

Sentiment-classification sandbox built on top of a lightweight transformer architecture. The project fine-tunes a generic transformers model on the IMDB reviews dataset using a configurable PyTorch training loop.

## Project layout

```
Transformers/
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
python train_sentiment.py
```

The script loads `configs.imdb:IMDBConfig` by default. To change hyperparameters, edit or subclass the dataclasses in `configs/` (see `configs/base.py` for shared defaults) and update the configuration target inside `train_sentiment.py`.

Loss curves and metrics are saved to the paths declared on the active config (`history_path`, `plot_path`). Legacy Neuscenes scripts (`scripts/` and `src/`) have been retired so the repository focuses solely on the IMDB sentiment workflow.
