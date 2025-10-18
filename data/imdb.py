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


# Function: _tokenize
# Description: Tokenize raw review text into word-level tokens using regex boundaries.
# Args:
#   text: Raw review string to tokenize.
# Returns:
#   List of token strings extracted from the input text.
def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text)


# Function: _normalise
# Description: Normalize a value by a denominator with zero-division protection.
# Args:
#   value: Numerator for the normalisation.
#   denominator: Denominator used to scale the numerator.
# Returns:
#   Normalised float result bounded at zero when denominator is invalid.
def _normalise(value: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return value / denominator


class IMDBDataset(Dataset):
    """Dataset wrapping IMDB reviews into transformer-ready feature tensors."""

    FEATURE_DIM = 4

    # Function: __init__
    # Description: Load the IMDB split and prepare token-based feature tensors.
    # Args:
    #   split: Dataset portion, expected to be 'train' or 'test'.
    #   max_tokens: Maximum number of tokens to encode per review.
    #   cache_dir: Optional path used by datasets library for caching.
    #   dataset_name: Dataset identifier for Hugging Face datasets.
    def __init__(
        self,
        split       : str,
        max_tokens  : int = 256,
        cache_dir   : Optional[Path] = None,
        dataset_name: str = "imdb",
    ) -> None:
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

    # Function: __len__
    # Description: Return the number of review samples available in the dataset.
    # Returns:
    #   Integer count of dataset elements.
    def __len__(self) -> int:
        return len(self.texts)

    # Function: _encode_token
    # Description: Convert a single token into a feature vector capturing character makeup.
    # Args:
    #   token: Word token to transform into numerical features.
    # Returns:
    #   Tensor encoding token statistics (length, alpha/digit ratios, bias).
    def _encode_token(self, token: str) -> torch.Tensor:
        length = len(token)
        alpha = sum(char.isalpha() for char in token)
        digits = sum(char.isdigit() for char in token)
        clipped_len = min(length, _MAX_TOKEN_LENGTH)
        length_norm = clipped_len / _MAX_TOKEN_LENGTH
        alpha_ratio = _normalise(alpha, length)
        digit_ratio = _normalise(digits, length)
        return torch.tensor([length_norm, alpha_ratio, digit_ratio, 1.0], dtype=torch.float32)

    # Function: __getitem__
    # Description: Build padded token feature tensor and sentiment label for an index.
    # Args:
    #   index: Sample index within the dataset.
    # Returns:
    #   Tuple containing tensorised token features and target label tensor.
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[index]
        label = float(self.labels[index])
        tokens = _tokenize(text)[: self.max_tokens]
        features = torch.zeros(self.max_tokens, self.FEATURE_DIM, dtype=torch.float32)
        for position, token in enumerate(tokens):
            features[position] = self._encode_token(token)
        target = torch.tensor([label], dtype=torch.float32)
        return features, target


# Function: build_imdb_dataloaders
# Description: Construct train and test dataloaders for IMDB sentiment classification.
# Args:
#   batch_size: Number of samples per batch.
#   max_tokens: Max tokens retained per review.
#   num_workers: Number of subprocesses used for data loading.
#   cache_dir: Location to cache downloaded dataset files.
#   dataset_name: Hugging Face dataset identifier.
# Returns:
#   Tuple containing training and test dataloaders.
def build_imdb_dataloaders(
    batch_size : int = 32,
    max_tokens : int = 256,
    num_workers: int = 0,
    cache_dir  : Optional[Path] = None,
    dataset_name: str = "imdb",
) -> Tuple[DataLoader, DataLoader]:
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
