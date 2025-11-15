from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .core import PositionalEncoding, TransformerEncoderLayer


@dataclass
class TransformersModelConfig:

    input_dim         : int
    embed_dim         : int           = 256
    depth             : int           = 6
    num_heads         : int           = 8
    mlp_ratio         : float         = 4.0
    dropout           : float         = 0.1
    attention_dropout : float         = 0.1
    use_cls_token     : bool          = True
    cls_head_dim      : Optional[int] = None
    num_outputs       : int           = 1
    pooling           : str           = "cls"
    vocab_size        : Optional[int] = None

    ''' Function: __post_init__
        Description: Validate configuration parameters after initialization.
        Args:
            None
        Returns:
            None
    '''
    def __post_init__(self) -> None:
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if self.vocab_size is None and self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive when provided.")


class TransformersModel(nn.Module):
    """Transformer backbone producing review representations."""
    ''' Function: __init__
        Description: Initialize transformer model with configuration.
        Args:
            config : Model configuration specifying architecture parameters.
        Returns:
            None
    '''
    def __init__(self, config: TransformersModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding: Optional[nn.Embedding] = None
        if config.vocab_size is not None:
            self.token_embedding = nn.Embedding(
                config.vocab_size,
                config.embed_dim,
                padding_idx=0,
            )
            self.input_proj = None
        else:
            self.input_proj = nn.Linear(config.input_dim, config.embed_dim)
        self.position   = PositionalEncoding(config.embed_dim,
                                             dropout = config.dropout,
                                            )
        self.encoder = nn.ModuleList(TransformerEncoderLayer(embed_dim         = config.embed_dim,
                                                             num_heads         = config.num_heads,
                                                             mlp_ratio         = config.mlp_ratio,
                                                             dropout           = config.dropout,
                                                             attention_dropout = config.attention_dropout,
                                                            )
            for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.embed_dim)

        if config.cls_head_dim:
            self.head = nn.Sequential(
                nn.Linear(config.embed_dim, config.cls_head_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.cls_head_dim, config.num_outputs),
            )
        else:
            self.head = nn.Linear(config.embed_dim, config.num_outputs)

        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)

        self.apply(self._init_weights)

    ''' Function: _init_weights
        Description: Initialise module weights following transformer conventions.
        Args:
            module : Neural network module to initialize.
        Returns:
            None
    '''
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    ''' Function: forward_features
        Description: Produce pooled latent features prior to the classification head.
        Args:
            inputs : Input token feature tensor.
            mask   : Optional attention mask for padding tokens.
        Returns:
            Pooled feature representation tensor.
    '''
    def forward_features(self,
                         inputs : Tensor,
                         attention_mask: Optional[Tensor] = None,
                        ) -> Tensor:
        """Produce pooled latent features prior to the classification head."""
        if self.token_embedding is not None:
            if inputs.dtype != torch.long:
                inputs = inputs.long()
            x = self.token_embedding(inputs)
            token_mask = attention_mask if attention_mask is not None else inputs.ne(0)
        else:
            x = self.input_proj(inputs)
            token_mask = attention_mask
        batch_size = x.size(0)

        if token_mask is not None:
            token_mask = token_mask.to(dtype=torch.bool)
        if self.config.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if token_mask is not None:
                cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=token_mask.device)
                token_mask = torch.cat([cls_mask, token_mask], dim=1)

        x = self.position(x)
        for layer in self.encoder:
            x = layer(x, mask=token_mask)
        x = self.norm(x)

        if self.config.pooling == "cls":
            return x[:, 0]

        if self.config.use_cls_token:
            tokens = x[:, 1:]
            mask_tokens = token_mask[:, 1:] if token_mask is not None else None
        else:
            tokens = x
            mask_tokens = token_mask

        if mask_tokens is not None:
            lengths = mask_tokens.sum(dim=1, keepdim=True).clamp(min=1)
            return (tokens * mask_tokens.unsqueeze(-1)).sum(dim=1) / lengths
        return tokens.mean(dim=1)

    ''' Function: forward
        Description: Execute the full model pipeline and return prediction logits.
        Args:
            inputs : Input token feature tensor.
            mask   : Optional attention mask for padding tokens.
        Returns:
            Output logits tensor.
    '''
    def forward(self,
                inputs : Tensor,
                attention_mask: Optional[Tensor] = None,
               ) -> Tensor:
        """Execute the full model pipeline and return prediction logits."""
        features = self.forward_features(inputs,
                                         attention_mask,
                                        )
        return self.head(features)
