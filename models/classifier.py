from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn

from transformer_core import PositionalEncoding, TransformerEncoderLayer


class ClassifierModel(nn.Module):
    """Transformer encoder backbone for sentiment classification."""

    def __init__(self, config: Any) -> None:
        super().__init__()

        embed_dim = int(getattr(config, "embed_dim", 0))
        depth = int(getattr(config, "depth", 0))
        num_outputs = int(getattr(config, "num_outputs", 0))

        if embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer.")
        if depth <= 0:
            raise ValueError("depth must be a positive integer.")
        if num_outputs <= 0:
            raise ValueError("num_outputs must be a positive integer.")

        self.config = config
        self.use_rope = bool(getattr(config, "use_rope", True))
        self.use_cls_token = bool(getattr(config, "use_cls_token", True))
        self.pooling = getattr(config, "pooling", "cls")

        self.token_embedding: Optional[nn.Embedding] = None
        self.input_proj: Optional[nn.Linear] = None

        vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is not None and int(vocab_size) > 0:
            self.token_embedding = nn.Embedding(int(vocab_size),
                                                embed_dim,
                                                padding_idx = 0)
        else:
            input_dim = getattr(config, "input_dim", None)
            if input_dim is None or int(input_dim) <= 0:
                raise ValueError("input_dim must be a positive integer when vocab_size is not set.")
            self.input_proj = nn.Linear(int(input_dim), embed_dim)

        self.position: Optional[PositionalEncoding] = None
        if not self.use_rope:
            self.position = PositionalEncoding(max_len   = getattr(config, "max_length", 512),
                                               embed_dim = embed_dim,
                                               dropout   = config.dropout,
                                               method    = "trainable")

        self.encoder = nn.ModuleList(TransformerEncoderLayer(config = config)
                                     for _ in range(depth)
                                    )
        self.norm = nn.LayerNorm(embed_dim)

        cls_head_dim = getattr(config, "cls_head_dim", None)
        if cls_head_dim:
            self.head = nn.Sequential(nn.Linear(embed_dim, int(cls_head_dim)),
                                      nn.GELU(),
                                      nn.Dropout(config.dropout),
                                      nn.Linear(int(cls_head_dim), num_outputs),
                                     )
        else:
            self.head = nn.Linear(embed_dim, num_outputs)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std = 0.02)
        else:
            self.register_parameter("cls_token", None)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise module weights following transformer conventions."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std = 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self,
                         inputs         : Tensor,
                         attention_mask : Optional[Tensor] = None,
                        ) -> Tensor:
        """Produce pooled encoder features before the classification head."""
        if self.token_embedding is not None:
            if inputs.dtype != torch.long:
                inputs = inputs.long()
            x = self.token_embedding(inputs)
            token_mask = attention_mask if attention_mask is not None else inputs.ne(0)
        else:
            if self.input_proj is None:
                raise RuntimeError("input_proj must be initialised when token_embedding is not used.")
            x = self.input_proj(inputs)
            token_mask = attention_mask

        batch_size = x.size(0)

        if token_mask is not None:
            token_mask = token_mask.to(dtype = torch.bool)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim = 1)
            if token_mask is not None:
                cls_mask = torch.ones(batch_size, 1, dtype = torch.bool, device = token_mask.device)
                token_mask = torch.cat([cls_mask, token_mask], dim = 1)

        if self.position is not None:
            x = self.position(x)

        for layer in self.encoder:
            x = layer(x, mask = token_mask)
        x = self.norm(x)

        if self.pooling == "cls":
            return x[:, 0]

        if self.use_cls_token:
            tokens = x[:, 1:]
            mask_tokens = token_mask[:, 1:] if token_mask is not None else None
        else:
            tokens = x
            mask_tokens = token_mask

        if mask_tokens is not None:
            lengths = mask_tokens.sum(dim = 1, keepdim = True).clamp(min = 1)
            return (tokens * mask_tokens.unsqueeze(-1)).sum(dim = 1) / lengths
        return tokens.mean(dim = 1)

    def forward(self,
                inputs         : Tensor,
                attention_mask : Optional[Tensor] = None,
               ) -> Tensor:
        """Execute the full classifier forward pass and return logits."""
        features = self.forward_features(inputs,
                                         attention_mask,
                                        )
        return self.head(features)
