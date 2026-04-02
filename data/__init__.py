"""Data utilities for the Transformers project."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .imdb import IMDBDataRead, DataPrep, Tokenize


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

    resolved_data_path = data_path or dataset_root or cache_dir or Path("data/imdb")
    prep = DataPrep(
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
    reader = IMDBDataRead(path=dataset_root or Path("data/imdb"), url_path=url_path)
    return reader.extract_data()


__all__ = [
    "IMDBDataRead",
    "DataPrep",
    "Tokenize",
    "build_imdb_dataloaders",
    "download_imdb_dataset",
]
