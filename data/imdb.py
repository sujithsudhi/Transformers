"""IMDB dataset loading and tokenization helpers."""

from __future__ import annotations

import logging
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

_TOKENIZER_CACHE: dict[str, Any] = {}


def _get_tokenizer(tokenizer_name: str) -> Any:
    tokenizer = _TOKENIZER_CACHE.get(tokenizer_name)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        _TOKENIZER_CACHE[tokenizer_name] = tokenizer
    return tokenizer


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


class IMDBDataRead:
    """Load and optionally download the extracted IMDB review dataset."""

    def __init__(self,
                 path: Path | str | None = None,
                 url_path: str = "",
                 download: bool = True,
                ) -> None:
        self.data_path = Path(path or "data/imdb")
        self.url_path = url_path
        self.download = download

    def _manifest_path(self, path: Path) -> Path:
        return path.parent / "aclImdb_manifest.pt"

    def _read_files(self, path: Path, pattern: str = "*.txt") -> dict[str, list[Any]]:
        reviews = {"texts": [], "label": []}
        file_list = sorted(path.glob(pattern))
        desc = f"Reading {path.parent.name}/{path.name}"

        for file_path in tqdm(file_list, desc=desc):
            label = 1 if file_path.parent.name == "pos" else 0
            text = file_path.read_text(encoding="utf-8")
            reviews["texts"].append(text)
            reviews["label"].append(label)

        return reviews

    def _load_from_local(self, path: Path) -> dict[str, dict[str, list[Any]]]:
        train_path = path / "train"
        test_path = path / "test"

        if not (train_path.exists() and test_path.exists() and train_path.is_dir() and test_path.is_dir()):
            raise FileNotFoundError(
                "Expected extracted IMDB folders at "
                f"'{path}'. Missing 'train/' or 'test/' subdirectories."
            )

        manifest_path = self._manifest_path(path)
        if manifest_path.exists():
            logger.info("Loading cached IMDB manifest from %s", manifest_path)
            return _torch_load(manifest_path)

        logger.info("Reading extracted IMDB reviews from %s", path)

        train_pos = self._read_files(path=train_path / "pos")
        train_neg = self._read_files(path=train_path / "neg")
        test_pos = self._read_files(path=test_path / "pos")
        test_neg = self._read_files(path=test_path / "neg")

        train_data = {
            "text": train_pos["texts"] + train_neg["texts"],
            "label": train_pos["label"] + train_neg["label"],
        }
        test_data = {
            "text": test_pos["texts"] + test_neg["texts"],
            "label": test_pos["label"] + test_neg["label"],
        }

        logger.info(
            "Train reviews: pos=%s neg=%s",
            len(train_pos["texts"]),
            len(train_neg["texts"]),
        )
        logger.info(
            "Test reviews: pos=%s neg=%s",
            len(test_pos["texts"]),
            len(test_neg["texts"]),
        )

        dataset = {"Train": train_data, "Test": test_data}
        torch.save(dataset, manifest_path)
        logger.info("Saved cached IMDB manifest to %s", manifest_path)
        return dataset

    def extract_data(self) -> dict[str, dict[str, list[Any]]]:
        data_path = self._download_dataset()
        return self._load_from_local(path=data_path)

    def _download_dataset(self) -> Path:
        self.data_path.mkdir(parents=True, exist_ok=True)

        tar_path = self.data_path / "aclImdb_v1.tar.gz"
        if not tar_path.exists():
            if not self.download:
                raise FileNotFoundError(
                    f"IMDB archive not found at '{tar_path}' and download is disabled."
                )
            if not self.url_path:
                raise ValueError(
                    "url_path must be set to download IMDB when the local archive is missing."
                )
            logger.info("Downloading IMDB dataset from %s to %s", self.url_path, tar_path)
            urllib.request.urlretrieve(self.url_path, tar_path)
        else:
            logger.info("IMDB archive already exists at %s", tar_path)

        extract_path = self.data_path / "aclImdb"
        if not extract_path.exists():
            logger.info("Extracting IMDB dataset to %s", extract_path)
            with tarfile.open(tar_path, "r:gz") as archive:
                archive.extractall(path=self.data_path)
        else:
            logger.info("IMDB dataset already extracted at %s", extract_path)

        return extract_path


class Tokenize(Dataset):
    """Dataset that stores raw texts and defers tokenization to the collate step."""

    def __init__(self,
                 texts: Sequence[str],
                 labels: Sequence[int],
                 tokenizer: Any,
                 max_length: int = 256,
                ) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Expose metadata expected by the trainer/model wiring.
        self.vocab_size = self.tokenizer.vocab_size
        self.feature_dim = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


class BatchTokenizer:
    """Collate raw reviews into batched token IDs and BCE targets."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self,
        batch: Sequence[tuple[str, int]],
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        texts = [text for text, _ in batch]
        labels = [label for _, label in batch]

        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        targets = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        return {
            "inputs": {
                "inputs": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
            "targets": targets,
        }


class DataPrep:
    """Prepare train/test dataloaders for IMDB sentiment experiments."""

    def __init__(self,
                 data_path: Path | str,
                 batch_size: int = 32,
                 num_workers: int = 8,
                 max_tokens: int = 256,
                 url_path: str = "",
                 tokenizer_name: str = "bert-base-uncased",
                 pin_memory: bool = True,
                 download: bool = True,
                ) -> None:
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_tokens = max_tokens
        self.url_path = url_path
        self.tokenizer_name = tokenizer_name
        self.pin_memory = pin_memory
        self.download = download

    def prep(self, split: str = "both"):
        reader = IMDBDataRead(
            path=self.data_path,
            url_path=self.url_path,
            download=self.download,
        )

        dataset_splits = reader.extract_data()
        tokenizer = _get_tokenizer(self.tokenizer_name)
        collate_fn = BatchTokenizer(tokenizer=tokenizer, max_length=self.max_tokens)

        train_dataset = Tokenize(
            texts=dataset_splits["Train"]["text"],
            labels=dataset_splits["Train"]["label"],
            tokenizer=tokenizer,
            max_length=self.max_tokens,
        )
        test_dataset = Tokenize(
            texts=dataset_splits["Test"]["text"],
            labels=dataset_splits["Test"]["label"],
            tokenizer=tokenizer,
            max_length=self.max_tokens,
        )

        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": False,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.num_workers > 0,
            "collate_fn": collate_fn,
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

        if split == "both":
            return train_loader, test_loader
        if split == "train":
            return train_loader
        if split == "test":
            return test_loader
        raise ValueError(f"Unsupported split '{split}'. Expected 'both', 'train', or 'test'.")

