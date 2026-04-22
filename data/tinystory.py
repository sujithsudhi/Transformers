"""TinyStories dataset loading, tokenization, and streaming helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        """
        Return the input iterable unchanged when tqdm is unavailable.
        Args:
            iterable : Iterable that would otherwise be wrapped by tqdm.
            *args    : Unused positional tqdm arguments kept for compatibility.
            **kwargs : Unused keyword tqdm arguments kept for compatibility.
        Returns:
            The original iterable without a progress bar wrapper.
        """
        del args, kwargs
        return iterable

logger = logging.getLogger(__name__)

_TOKENIZER_CACHE: dict[str, GPT2TokenizerFast] = {}
_LOCAL_FILE_BUILDERS = {".txt"    : "text",
                        ".text"   : "text",
                        ".json"   : "json",
                        ".jsonl"  : "json",
                        ".parquet": "parquet",
                       }
_LOCAL_SPLIT_ALIASES = {"train"     : ("train",),
                        "validation": ("validation", "valid", "val"),
                       }


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

    def __init__(self,
                 dataset      : Optional[str] = "roneneldan/TinyStories",
                 data_path    : Path | str | None = None,
                 dataset_root : Path | str | None = None,
                ) -> None:
        self.dataset = dataset or "roneneldan/TinyStories"
        self.data_path = Path(data_path).expanduser() if data_path else None
        self.dataset_root = Path(dataset_root).expanduser() if dataset_root else None
        self.source_id = self.dataset

    def _candidate_local_paths(self) -> list[Path]:
        """
        Collect unique local paths that may contain TinyStories data.
        Returns:
            Ordered list of existing local dataset candidates.
        """
        candidates: list[Path] = []
        for raw_candidate in (self.data_path, self.dataset_root):
            if raw_candidate is None or not raw_candidate.exists():
                continue
            resolved = raw_candidate.resolve()
            if resolved not in candidates:
                candidates.append(resolved)

        dataset_path = Path(self.dataset).expanduser()
        if dataset_path.exists():
            resolved = dataset_path.resolve()
            if resolved not in candidates:
                candidates.append(resolved)
        return candidates

    def _is_saved_dataset_path(self,
                               path : Path,
                              ) -> bool:
        """
        Check whether a path looks like a Hugging Face dataset saved to disk.
        Args:
            path : Candidate local dataset directory.
        Returns:
            True when the directory exposes dataset save markers.
        """
        if not path.is_dir():
            return False

        marker_paths = (path / "dataset_dict.json",
                        path / "state.json")
        return any(marker.exists() for marker in marker_paths)

    def resolve_source_id(self) -> str:
        """
        Resolve the cache/source identifier for the active TinyStories data source.
        Returns:
            Stable source identifier used for token-cache names.
        """
        for path in self._candidate_local_paths():
            if self._is_saved_dataset_path(path) or self._resolve_local_data_files(path) is not None:
                return path.as_posix()
        return self.dataset

    def _load_saved_dataset(self,
                            path : Path,
                           ) -> Optional[DatasetDict]:
        """
        Load a previously saved Hugging Face dataset from disk.
        Args:
            path : Candidate local dataset directory.
        Returns:
            DatasetDict with train/validation splits, or `None` when unsupported.
        """
        if not path.is_dir():
            return None

        if not self._is_saved_dataset_path(path):
            return None

        dataset = load_from_disk(path.as_posix())
        if isinstance(dataset, DatasetDict):
            return dataset

        logger.warning("Ignoring local TinyStories dataset at %s because it does not expose train/validation splits.",
                       path)
        return None

    def _resolve_local_data_files(self,
                                  path : Path,
                                 ) -> Optional[tuple[str, dict[str, list[str]]]]:
        """
        Resolve local TinyStories split files into a load_dataset builder payload.
        Args:
            path : Candidate local dataset directory.
        Returns:
            Tuple of builder name and data_files mapping, or `None` when no supported layout is found.
        """
        if not path.is_dir():
            return None

        split_files: dict[str, list[Path]] = {}
        for split_name, aliases in _LOCAL_SPLIT_ALIASES.items():
            matched_files: list[Path] = []
            for alias in aliases:
                split_dir = path / alias
                for suffix in _LOCAL_FILE_BUILDERS:
                    direct_file = path / f"{alias}{suffix}"
                    if direct_file.is_file():
                        matched_files.append(direct_file)
                    if split_dir.is_dir():
                        matched_files.extend(
                            sorted(file_path for file_path in split_dir.rglob(f"*{suffix}") if file_path.is_file())
                        )

            unique_files = list(dict.fromkeys(file_path.resolve() for file_path in matched_files))
            if unique_files:
                split_files[split_name] = unique_files

        if "train" not in split_files or "validation" not in split_files:
            return None

        suffixes = {file_path.suffix.lower()
                    for files in split_files.values()
                    for file_path in files
                   }
        if len(suffixes) != 1:
            logger.warning("Ignoring local TinyStories dataset at %s because split files use mixed formats: %s",
                           path,
                           sorted(suffixes))
            return None

        suffix = next(iter(suffixes))
        builder_name = _LOCAL_FILE_BUILDERS.get(suffix)
        if builder_name is None:
            logger.warning("Ignoring local TinyStories dataset at %s because %s files are unsupported.",
                           path,
                           suffix)
            return None

        data_files = {split_name: [file_path.as_posix() for file_path in files]
                      for split_name, files in split_files.items()
                     }
        return builder_name, data_files

    def _load_local_dataset(self) -> Optional[DatasetDict]:
        """
        Load TinyStories from a supported local source when available.
        Returns:
            DatasetDict loaded from disk or local split files, otherwise `None`.
        """
        for path in self._candidate_local_paths():
            dataset = self._load_saved_dataset(path)
            if dataset is not None:
                self.source_id = path.as_posix()
                logger.info("Loading TinyStories dataset from saved dataset at %s", path)
                return dataset

            resolved_files = self._resolve_local_data_files(path)
            if resolved_files is None:
                continue

            builder_name, data_files = resolved_files
            self.source_id = path.as_posix()
            logger.info("Loading TinyStories dataset from local %s files at %s", builder_name, path)
            return load_dataset(builder_name, data_files = data_files)

    def load_dataset(self):
        dataset = self._load_local_dataset()
        if dataset is None:
            self.source_id = self.dataset
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
                 dataset      : str = "roneneldan/TinyStories",
                 block_size   : int = 256,
                 batch_size   : int = 32,
                 shuffle      : bool = True,
                 num_workers  : int = 8,
                 pin_memory   : bool = True,
                 cache_dir    : Optional[str] = "data/cache/tinystories",
                 tokenizer_name: str = "gpt2",
                 stride       : Optional[int] = None,
                 use_map      : bool = False,
                 map_num_proc : int = 8,
                 map_batch_size: int = 1000,
                 data_path    : Path | str | None = None,
                 dataset_root : Path | str | None = None,
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
        self.data_path = Path(data_path).expanduser() if data_path else None
        self.dataset_root = Path(dataset_root).expanduser() if dataset_root else None

        # Backwards-compatible attribute aliases.
        self.blockSize = self.block_size
        self.batchSize = self.batch_size
        self.numWorkers = self.num_workers
        self.cacheDir = self.cache_dir
        self.mapNumProc = self.map_num_proc
        self.mapBatchSize = self.map_batch_size

    def _cache_source_id(self) -> str:
        """
        Resolve the dataset identifier used when naming token caches.
        Returns:
            Source identifier matching the dataset selection logic.
        """
        reader = DataRead(dataset      = self.dataset,
                          data_path    = self.data_path,
                          dataset_root = self.dataset_root)
        return reader.resolve_source_id()

    def _cache_path(self,
                    split_name     : str,
                    tokenizer_name : str,
                    dataset_id     : Optional[str] = None,
                   ) -> Optional[Path]:
        """
        Build the on-disk cache path for one tokenized split.
        Args:
            split_name     : Dataset split name such as `train` or `validation`.
            tokenizer_name : Tokenizer name used to create the cache.
            dataset_id     : Optional explicit dataset identifier from the active reader.
        Returns:
            Cache file path, or `None` when caching is disabled.
        """
        if self.cache_dir is None:
            return None
        source_id = dataset_id or self._cache_source_id()
        safe_dataset = source_id.replace(":", "_").replace("\\", "_").replace("/", "_")
        safe_tokenizer = tokenizer_name.replace("/", "_")
        filename = f"{safe_dataset}_{split_name}_{safe_tokenizer}.pt"
        return self.cache_dir / filename

    def prep(self):
        reader = DataRead(dataset      = self.dataset,
                          data_path    = self.data_path,
                          dataset_root = self.dataset_root)
        dataset = reader.load_dataset()

        train_data = dataset["train"]
        val_data = dataset["validation"]

        tokenizer = Tokenizer(
            max_tokens=self.block_size,
            tokenizer_name=self.tokenizer_name,
        )
        train_cache_path = self._cache_path("train",
                                            tokenizer.tokenizer_name,
                                            dataset_id = reader.source_id)
        val_cache_path = self._cache_path("validation",
                                          tokenizer.tokenizer_name,
                                          dataset_id = reader.source_id)

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
