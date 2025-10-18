"""Embedding modules used by the foundation transformer."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """Lookup embedding layer with optional padding index."""

    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embed_dim,
                                      padding_idx=padding_idx,)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with optional dropout."""

    def __init__(self, embed_dim: int, max_len: int = 10000, dropout: float = 0.0) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_table", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        length = x.size(1)
        positional = self.positional_table[:, offset : offset + length]
        return self.dropout(x + positional)

