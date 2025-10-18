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
python train_sentiment.py \
  --batch-size 32 \
  --epochs 5 \
  --max-tokens 256
```

By default training artefacts are written to `results/history.json` and `results/history.png`. The script prints final test-set loss metrics once training completes.
