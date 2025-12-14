"""
Data utilities for the Transformers sentiment project.
"""

from .imdb import IMDBDataRead, DataPrep, Tokenize

__all__ = ["IMDBDataset", "build_imdb_dataloaders", "download_imdb_dataset"]

# Scene dataset utilities are optional; import when available.
try:
    from data.neuscene import (
        DatasetMetadata,
        SceneDatasetConfig,
        SceneRecord,
        SensorSummary,
        load_scene_metadata,
        split_scene_dataset,
    )
except ModuleNotFoundError:
    DatasetMetadata = SceneDatasetConfig = SceneRecord = SensorSummary = None  # type: ignore
else:
    __all__ += [
        "DatasetMetadata",
        "SceneDatasetConfig",
        "SceneRecord",
        "SensorSummary",
        "load_scene_metadata",
        "split_scene_dataset",
    ]
