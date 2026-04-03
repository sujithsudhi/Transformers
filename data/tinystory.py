"""TinyStories dataset loading, tokenization, and streaming helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2TokenizerFast

logger = logging.getLogger(__name__)

_TOKENIZER_CACHE: dict[str, GPT2TokenizerFast] = {}


def _get_tokenizer(tokenizer_name: str) -> GPT2TokenizerFast:
    tokenizer = _TOKENIZER_CACHE.get(tokenizer_name)
    if tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _TOKENIZER_CACHE[tokenizer_name] = tokenizer
    return tokenizer


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _encode_texts(tokenizer: GPT2TokenizerFast, texts: Sequence[str]) -> dict[str, list[list[int]]]:
    # TinyStories preprocessing tokenizes whole documents first, then windows them
    # into fixed-length training chunks later. The HF warning about 1024-token inputs
    # is meant for direct model calls, so we suppress it here to avoid noisy logs.
    return tokenizer(texts, add_special_tokens=False, truncation=False, verbose=False)


def _tokenize_batch_texts(batch: dict[str, list[str]], tokenizer_name: str) -> dict[str, list[list[int]]]:
    tokenizer = _get_tokenizer(tokenizer_name)
    return _encode_texts(tokenizer, batch["text"])


class DataRead:
    """Load TinyStories splits from Hugging Face datasets."""

    def __init__(self, dataset: Optional[str] = "roneneldan/TinyStories") -> None:
        self.dataset = dataset or "roneneldan/TinyStories"

    def load_dataset(self):
        dataset = load_dataset(self.dataset)
        train_split = dataset["train"]
        val_split = dataset["validation"]

        logger.info("Number of train samples: %s", len(train_split))
        logger.info("Number of validation samples: %s", len(val_split))
        logger.info("Data schema: %s", train_split.features)

        if len(train_split) > 0:
            logger.info("Sample data: %s", train_split[0]["text"])

        return dataset

    # Backwards-compatible alias.
    def loadDataset(self):
        return self.load_dataset()


class Tokenizer:
    """Tokenize TinyStories documents into contiguous token streams."""

    def __init__(self,
                 max_tokens: int = 1024,
                 tokenizer_name: str = "gpt2",
                ) -> None:
        self.max_tokens = max_tokens
        self.tokenizer_name = tokenizer_name
        self.tokenizer = _get_tokenizer(tokenizer_name)
        self.eos = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size

        # Backwards-compatible attribute aliases.
        self.maxTokens = self.max_tokens
        self.tokenizerName = self.tokenizer_name

        logger.info("Vocabulary size: %s", self.tokenizer.vocab_size)

    def tokenize_split(self,
                       split,
                       cache_path: Optional[Path] = None,
                       batch_size: int = 1000,
                      ) -> list[int]:
        if cache_path is not None and cache_path.exists():
            logger.info("Loading cached tokens from %s", cache_path)
            return _torch_load(cache_path)

        tokens: list[int] = []
        texts = split["text"]
        iterator = tqdm(range(0, len(texts), batch_size), desc="Tokenizing", leave=False)

        for start in iterator:
            batch = texts[start : start + batch_size]
            encoded = _encode_texts(self.tokenizer, batch)

            for ids in encoded["input_ids"]:
                tokens.extend(ids)
                tokens.append(self.eos)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tokens, cache_path)
            logger.info("Saved cached tokens to %s", cache_path)

        return tokens

    def tokenize_split_map(self,
                           split,
                           cache_path: Optional[Path] = None,
                           batch_size: int = 1000,
                           num_proc: int = 8,
                          ) -> list[int]:
        if cache_path is not None and cache_path.exists():
            logger.info("Loading cached tokens from %s", cache_path)
            return _torch_load(cache_path)

        tokenized = split.map(
            _tokenize_batch_texts,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            remove_columns=["text"],
            fn_kwargs={"tokenizer_name": self.tokenizer_name},
        )

        tokens: list[int] = []
        for ids in tokenized["input_ids"]:
            tokens.extend(ids)
            tokens.append(self.eos)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tokens, cache_path)
            logger.info("Saved cached tokens to %s", cache_path)

        return tokens

    # Backwards-compatible aliases.
    def tokenizeSplit(self, split, cache_path: Optional[Path] = None, batch_size: int = 1000):
        return self.tokenize_split(split, cache_path=cache_path, batch_size=batch_size)

    def tokenizeSplitMap(
        self,
        split,
        cache_path: Optional[Path] = None,
        batch_size: int = 1000,
        num_proc: int = 8,
    ):
        return self.tokenize_split_map(
            split,
            cache_path=cache_path,
            batch_size=batch_size,
            num_proc=num_proc,
        )


class DataStreamer(Dataset):
    """Slice a contiguous token stream into autoregressive training windows."""

    def __init__(self, tokens: Sequence[int], blockSize: int, stride: int) -> None:
        super().__init__()
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = blockSize
        self.stride = max(1, stride)

        # Backwards-compatible attribute alias.
        self.blockSize = self.block_size

    def __len__(self) -> int:
        usable = len(self.tokens) - (self.block_size + 1)
        if usable < 0:
            return 0
        return (usable // self.stride) + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        chunk = self.tokens[start : start + self.block_size + 1]
        inputs = chunk[:-1]
        targets = chunk[1:]
        return inputs, targets


class DataPrep:
    """Prepare TinyStories dataloaders and tokenizer state."""

    def __init__(self,
                 dataset: str = "roneneldan/TinyStories",
                 block_size: int = 256,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 cache_dir: Optional[str] = "data/cache/tinystories",
                 tokenizer_name: str = "gpt2",
                 stride: Optional[int] = None,
                 use_map: bool = False,
                 map_num_proc: int = 8,
                 map_batch_size: int = 1000,
                ) -> None:
        self.dataset = dataset
        self.block_size = block_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.tokenizer_name = tokenizer_name
        self.stride = block_size if stride is None else max(1, stride)
        self.use_map = use_map
        self.map_num_proc = map_num_proc
        self.map_batch_size = map_batch_size

        # Backwards-compatible attribute aliases.
        self.blockSize = self.block_size
        self.batchSize = self.batch_size
        self.numWorkers = self.num_workers
        self.cacheDir = self.cache_dir
        self.mapNumProc = self.map_num_proc
        self.mapBatchSize = self.map_batch_size

    def _cache_path(self, split_name: str, tokenizer_name: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe_dataset = self.dataset.replace("/", "_")
        safe_tokenizer = tokenizer_name.replace("/", "_")
        filename = f"{safe_dataset}_{split_name}_{safe_tokenizer}.pt"
        return self.cache_dir / filename

    def prep(self):
        reader = DataRead(dataset=self.dataset)
        dataset = reader.load_dataset()

        train_data = dataset["train"]
        val_data = dataset["validation"]

        tokenizer = Tokenizer(
            max_tokens=self.block_size,
            tokenizer_name=self.tokenizer_name,
        )
        train_cache_path = self._cache_path("train", tokenizer.tokenizer_name)
        val_cache_path = self._cache_path("validation", tokenizer.tokenizer_name)

        if self.use_map:
            train_tokens = tokenizer.tokenize_split_map(
                train_data,
                cache_path=train_cache_path,
                batch_size=self.map_batch_size,
                num_proc=self.map_num_proc,
            )
            val_tokens = tokenizer.tokenize_split_map(
                val_data,
                cache_path=val_cache_path,
                batch_size=self.map_batch_size,
                num_proc=self.map_num_proc,
            )
        else:
            train_tokens = tokenizer.tokenize_split(
                train_data,
                cache_path=train_cache_path,
                batch_size=self.map_batch_size,
            )
            val_tokens = tokenizer.tokenize_split(
                val_data,
                cache_path=val_cache_path,
                batch_size=self.map_batch_size,
            )

        train_dataset = DataStreamer(
            tokens=train_tokens,
            blockSize=self.block_size,
            stride=self.stride,
        )
        val_dataset = DataStreamer(
            tokens=val_tokens,
            blockSize=self.block_size,
            stride=self.stride,
        )

        loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.num_workers > 0,
        }
        train_loader = DataLoader(train_dataset, shuffle=self.shuffle, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        return train_loader, val_loader, tokenizer

