"""IMDB dataset utilities for sentiment classification."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "The 'datasets' package is required for IMDB utilities. Install it via 'pip install datasets'."
    ) from exc


_TOKEN_PATTERN = re.compile(r"\b\w+\b")
_MAX_TOKEN_LENGTH = 20


def _tokenize(text: str) -> List[str]:
    """Tokenize a review into word-level tokens using regex boundaries."""
    return _TOKEN_PATTERN.findall(text)


def _normalise(value: float, denominator: float) -> float:
    """Normalise a value by its denominator, guarding against division by zero."""
    if denominator <= 0:
        return 0.0
    return value / denominator


def _load_local_split(path: Path) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            texts.append(str(payload["text"]))
            labels.append(int(payload["label"]))
    return texts, labels


def download_imdb_dataset(
    target_dir: Path = Path("data/imdb"),
    dataset_name: str = "imdb",
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Download the IMDB dataset and materialise JSONL splits under ``target_dir``."""
    target = target_dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    expected_files = {split: target / f"{split}.jsonl" for split in ("train", "test")}
    if not overwrite and all(path.exists() for path in expected_files.values()):
        return expected_files

    dataset = load_dataset(dataset_name)
    created: Dict[str, Path] = {}
    for split, path in expected_files.items():
        records = dataset[split]
        with path.open("w", encoding="utf-8") as handle:
            for text, label in zip(records["text"], records["label"]):
                json.dump({"text": text, "label": int(label)}, handle)
                handle.write("\n")
        created[split] = path
    return created


class IMDBDataset(Dataset):
    """Dataset wrapping IMDB reviews into transformer-ready feature tensors."""

    FEATURE_DIM = 4

    def __init__(
        self,
        split       : str,
        max_tokens  : int = 256,
        cache_dir   : Optional[Path] = None,
        dataset_name: str = "imdb",
        dataset_root: Optional[Path] = None,
        download    : bool = True,
    ) -> None:
        """Load an IMDB split and pre-compute token based features."""
        if split not in {"train", "test"}:
            raise ValueError("IMDBDataset split must be either 'train' or 'test'.")

        self.max_tokens = int(max_tokens)
        self.feature_dim = self.FEATURE_DIM

        local_texts: List[str] | None = None
        local_labels: List[int] | None = None

        root = dataset_root.expanduser().resolve() if dataset_root else None
        if root is not None:
            split_path = root / f"{split}.jsonl"
            if split_path.exists():
                local_texts, local_labels = _load_local_split(split_path)
            elif download:
                download_imdb_dataset(root, dataset_name=dataset_name, overwrite=False)
                if split_path.exists():
                    local_texts, local_labels = _load_local_split(split_path)

        if local_texts is None or local_labels is None:
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            self.texts = list(dataset["text"])
            self.labels = list(dataset["label"])
        else:
            self.texts = local_texts
            self.labels = local_labels
        self.max_tokens = int(max_tokens)

    def __len__(self) -> int:
        """Return the number of review samples available in the dataset."""
        return len(self.texts)

    def _encode_token(self, token: str) -> torch.Tensor:
        """Convert a token into a feature vector capturing character statistics."""
        length = len(token)
        alpha = sum(char.isalpha() for char in token)
        digits = sum(char.isdigit() for char in token)
        clipped_len = min(length, _MAX_TOKEN_LENGTH)
        length_norm = clipped_len / _MAX_TOKEN_LENGTH
        alpha_ratio = _normalise(alpha, length)
        digit_ratio = _normalise(digits, length)
        return torch.tensor([length_norm, alpha_ratio, digit_ratio, 1.0], dtype=torch.float32)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return padded token features and binary sentiment label for the given index."""
        text = self.texts[index]
        label = float(self.labels[index])
        tokens = _tokenize(text)[: self.max_tokens]
        features = torch.zeros(self.max_tokens, self.FEATURE_DIM, dtype=torch.float32)
        for position, token in enumerate(tokens):
            features[position] = self._encode_token(token)
        target = torch.tensor([label], dtype=torch.float32)
        return features, target


def build_imdb_dataloaders(
    batch_size : int = 32,
    max_tokens : int = 256,
    num_workers: int = 0,
    cache_dir  : Optional[Path] = None,
    dataset_name: str = "imdb",
    dataset_root: Optional[Path] = None,
    download    : bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Construct train and test dataloaders for IMDB sentiment classification."""
    cache_dir = cache_dir or Path("data/cache")
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if dataset_root is not None and download:
        download_imdb_dataset(dataset_root, dataset_name=dataset_name, overwrite=False)

    # Prepare dataset instances for each split with shared configuration.
    train_dataset = IMDBDataset(split       = "train",
                                max_tokens  = max_tokens,
                                cache_dir   = cache_dir,
                                dataset_name= dataset_name,
                                dataset_root= dataset_root,
                                download    = download,
                               )
    test_dataset = IMDBDataset(split       = "test",
                               max_tokens  = max_tokens,
                               cache_dir   = cache_dir,
                               dataset_name= dataset_name,
                               dataset_root= dataset_root,
                               download    = download,
                              )

    train_loader = DataLoader(dataset    = train_dataset,
                              batch_size = batch_size,
                              shuffle    = True,
                              num_workers= num_workers,
                              drop_last  = False,
                             )
    test_loader = DataLoader(dataset    = test_dataset,
                             batch_size = batch_size,
                             shuffle    = False,
                             num_workers= num_workers,
                             drop_last  = False,
                            )
    return train_loader, test_loader
