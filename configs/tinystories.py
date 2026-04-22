"""TinyStories language-model configuration leveraging shared defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .base import (AppConfig,
                   BaseDataConfig,
                   BaseDataloaderConfig,
                   BaseLossConfig,
                   BaseOptimizerConfig,
                   BaseTrainingConfig,
                  )
from .transformers import TransformersModelConfig


@dataclass(frozen=True)
class TinyStoriesDataConfig(BaseDataConfig):
    """Dataset and preprocessing parameters for TinyStories."""

    data_path      : Path          = Path(r"D:\Datasets\Basics\TinyStories")
    cache_dir      : Path          = Path("data/cache/tinystories")
    max_tokens     : int           = 512
    dataset_name   : str           = "roneneldan/TinyStories"
    dataset_root   : Path          = Path("data/tinystories")
    url_path       : Optional[str] = None
    stride         : Optional[int] = None
    use_map        : bool          = True
    map_num_proc   : int           = 8
    map_batch_size : int           = 1000


@dataclass(frozen=True)
class TinyStoriesModelConfig(TransformersModelConfig):
    """Decoder-model hyperparameters tuned for TinyStories language modeling."""

    embed_dim      : int   = 512
    depth          : int   = 4
    num_heads      : int   = 8
    mlp_ratio      : float = 2.0
    dropout        : float = 0.1
    max_length     : int   = 512
    input_dim      : int   = 0
    vocab_size     : int   = 0
    use_flash_attn : bool  = True
    use_rope       : bool  = True
    attention_type : str   = "global"


@dataclass(frozen=True)
class TinyStoriesTrainingConfig(BaseTrainingConfig):
    """Training loop defaults for TinyStories language modeling."""

    epochs                  : int        = 300
    lr                      : float      = 3e-4
    weight_decay            : float      = 0.01
    use_amp                 : str | bool = True
    amp_dtype               : str        = "fp16"
    early_stopping_patience : int | None = 10
    lr_reduction_patience   : int | None = None
    warmup_epochs           : int        = 5
    warmup_start_factor     : float      = 0.1
    use_cosine_decay        : bool       = True
    min_lr                  : float      = 1e-5


@dataclass(frozen=True)
class TinyStoriesDataloaderConfig(BaseDataloaderConfig):
    """Torch DataLoader parameters for TinyStories token streams."""

    batch_size  : int  = 64
    num_workers : int  = 4
    pin_memory  : bool = True


@dataclass(frozen=True)
class TinyStoriesOptimizerConfig(BaseOptimizerConfig):
    """Optimizer configuration tuned for decoder language modeling."""

    name         : str               = "adamw"
    lr           : float             = 3e-4
    weight_decay : float             = 0.1
    betas        : tuple[float, float] = (0.9, 0.95)
    eps          : float             = 1e-8


@dataclass(frozen=True)
class TinyStoriesLossConfig(BaseLossConfig):
    """Loss specification for next-token prediction."""

    name : str = "crossentropyloss"

    def as_dict(self) -> Dict[str, object]:
        return {"name" : self.name}


@dataclass(frozen=True)
class TinyStoriesConfig(AppConfig):
    """Application config describing TinyStories language-model training."""

    name            : str                         = "tinystories"
    data            : TinyStoriesDataConfig       = field(default_factory=TinyStoriesDataConfig)
    model           : TinyStoriesModelConfig      = field(default_factory=TinyStoriesModelConfig)
    training        : TinyStoriesTrainingConfig   = field(default_factory=TinyStoriesTrainingConfig)
    dataloader      : TinyStoriesDataloaderConfig = field(default_factory=TinyStoriesDataloaderConfig)
    optimizer       : TinyStoriesOptimizerConfig  = field(default_factory=TinyStoriesOptimizerConfig)
    loss            : TinyStoriesLossConfig       = field(default_factory=TinyStoriesLossConfig)
    history_path    : Path                        = Path("results/tiny_stories_history.json")
    plot_path       : Path                        = Path("results/tiny_stories_history.png")
    checkpoint_path : Path                        = Path("results/tiny_stories_transformer.pt")
    wandb_disabled  : bool                        = False
    wandb_project   : str                         = "transformers-tinystories"
    wandb_run_name  : str                         = "RoPE-Depth-4-Embed-512-Flash-AdamW-3e-4"
    tokenizer_name  : str                         = "gpt2"
