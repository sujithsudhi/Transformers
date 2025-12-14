"""Configuration package exposing base and application-specific classes."""

from .base import (
    AppConfig,
    BaseDataConfig,
    BaseDatasetConfig,
    BaseDataloaderConfig,
    BaseLossConfig,
    BaseModelConfig,
    BaseOptimizerConfig,
    BaseTrainingConfig,
)

__all__ = [
    "AppConfig",
    "BaseDataConfig",
    "BaseDatasetConfig",
    "BaseDataloaderConfig",
    "BaseLossConfig",
    "BaseModelConfig",
    "BaseOptimizerConfig",
    "BaseTrainingConfig",
]
try:
    from .transformers import (
        TransformersConfig,
        TransformersDataConfig,
        TransformersDataloaderConfig,
        TransformersLossConfig,
        TransformersModelConfig,
        TransformersOptimizerConfig,
        TransformersTrainingConfig,
    )
except ModuleNotFoundError:
    TransformersConfig = (
        TransformersDataConfig
    ) = TransformersDataloaderConfig = TransformersLossConfig = TransformersModelConfig = (
        TransformersOptimizerConfig
    ) = TransformersTrainingConfig = None  # type: ignore
else:
    __all__.extend(
        [
            "TransformersConfig",
            "TransformersDataConfig",
            "TransformersDataloaderConfig",
            "TransformersLossConfig",
            "TransformersModelConfig",
            "TransformersOptimizerConfig",
            "TransformersTrainingConfig",
        ]
    )
from .imdb import IMDBConfig, IMDBDataConfig, IMDBModelConfig, IMDBTrainingConfig

__all__.extend(
    [
        "IMDBConfig",
        "IMDBDataConfig",
        "IMDBModelConfig",
        "IMDBTrainingConfig",
    ]
)
