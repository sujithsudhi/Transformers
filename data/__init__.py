"""Data utilities for the Transformers project."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Optional


_LAZY_EXPORTS = {"IMDBDataRead"          : (".imdb", "IMDBDataRead"),
                 "IMDBDataPrep"          : (".imdb", "DataPrep"),
                 "IMDBTokenizedDataset"  : (".imdb", "Tokenize"),
                 "Tokenize"              : (".imdb", "Tokenize"),
                 "DataPrep"              : (".imdb", "DataPrep"),
                 "TinyStoriesDataPrep"   : (".tinystory", "DataPrep"),
                 "TinyStoriesDataRead"   : (".tinystory", "DataRead"),
                 "TinyStoriesTokenizer"  : (".tinystory", "Tokenizer"),
                 "DataStreamer"          : (".tinystory", "DataStreamer"),
                }


def _resolve_export(name: str) -> Any:
    """
    Resolve one lazily imported data export.
    Args:
        name : Public attribute requested from the data package.
    Returns:
        Resolved class or function object cached in module globals.
    Raises:
        AttributeError : Raised when the requested export is unsupported.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    """
    Lazily expose dataset helpers so one task does not import another task's stack.
    Args:
        name : Public attribute requested from this package.
    Returns:
        Lazily loaded export from the matching submodule.
    Raises:
        AttributeError : Raised when the attribute is not a known export.
    """
    return _resolve_export(name)


def build_imdb_dataloaders(
    *,
    batch_size: int = 32,
    max_tokens: int = 256,
    num_workers: int = 0,
    data_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    dataset_name: str = "imdb",
    dataset_root: Optional[Path] = None,
    url_path: str = "",
    tokenizer_name: str = "bert-base-uncased",
    pin_memory: bool = True,
    download: bool = True,
):
    if dataset_name.lower() != "imdb":
        raise ValueError(f"Unsupported dataset_name for IMDB loader helper: {dataset_name}")

    IMDBDataPrep = _resolve_export("IMDBDataPrep")
    resolved_data_path = data_path or dataset_root or cache_dir or Path("data/imdb")
    prep = IMDBDataPrep(
        data_path=resolved_data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_tokens=max_tokens,
        url_path=url_path,
        tokenizer_name=tokenizer_name,
        pin_memory=pin_memory,
        download=download,
    )
    return prep.prep()


def download_imdb_dataset(
    *,
    dataset_root: Optional[Path] = None,
    url_path: str = "",
):
    IMDBDataRead = _resolve_export("IMDBDataRead")
    reader = IMDBDataRead(path=dataset_root or Path("data/imdb"), url_path=url_path)
    return reader.extract_data()


__all__ = [
    "IMDBDataRead",
    "DataPrep",
    "IMDBDataPrep",
    "IMDBTokenizedDataset",
    "Tokenize",
    "TinyStoriesDataPrep",
    "TinyStoriesDataRead",
    "TinyStoriesTokenizer",
    "DataStreamer",
    "build_imdb_dataloaders",
    "download_imdb_dataset",
]
