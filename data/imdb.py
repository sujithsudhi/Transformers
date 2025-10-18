"""IMDB dataset utilities for sentiment classification."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


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


class IMDBDataset(Dataset):
    """Dataset wrapping IMDB reviews into transformer-ready feature tensors."""

    FEATURE_DIM = 4

    def __init__(
        self,
        split       : str,
        max_tokens  : int = 256,
        cache_dir   : Optional[Path] = None,
        dataset_name: str = "imdb",
    ) -> None:
        """Load an IMDB split and pre-compute token based features."""
        if split not in {"train", "test"}:
            raise ValueError("IMDBDataset split must be either 'train' or 'test'.")
        dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        self.texts: List[str] = list(dataset["text"])
        self.labels: List[int] = list(dataset["label"])
        self.max_tokens = int(max_tokens)
        self.feature_dim = self.FEATURE_DIM

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
) -> Tuple[DataLoader, DataLoader]:
    """Construct train and test dataloaders for IMDB sentiment classification."""
    cache_dir = cache_dir or Path("data/cache")
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataset instances for each split with shared configuration.
    train_dataset = IMDBDataset(split       = "train",
                                max_tokens  = max_tokens,
                                cache_dir   = cache_dir,
                                dataset_name= dataset_name,
                               )
    test_dataset = IMDBDataset(split       = "test",
                               max_tokens  = max_tokens,
                               cache_dir   = cache_dir,
                               dataset_name= dataset_name,
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
