from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = activation or nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        out_dim = output_dim or input_dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, module: nn.Module, dropout: float = 0.0) -> None:
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x + self.dropout(self.module(x, *args, **kwargs))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _reshape(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        return (
            x.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, seq_len, self.head_dim)
        )

    def _combine(self, x: Tensor, batch_size: int) -> Tensor:
        seq_len = x.size(1)
        return (
            x.view(batch_size, self.num_heads, seq_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = x.size(0)

        q = self._reshape(self.q_proj(x))
        k = self._reshape(self.k_proj(x))
        v = self._reshape(self.v_proj(x))

        attn_mask = None
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask > 0
            if mask.dim() == 2:
                mask = mask.unsqueeze(1) & mask.unsqueeze(2)
                mask = mask.repeat_interleave(self.num_heads, dim=0)
            elif mask.dim() == 3:
                if mask.size(0) == batch_size:
                    mask = mask.repeat_interleave(self.num_heads, dim=0)
                elif mask.size(0) != batch_size * self.num_heads:
                    raise ValueError("Attention mask batch dimension mismatch.")
            else:
                raise ValueError("Unsupported attention mask rank.")
            attn_mask = mask.to(q.device)

        attn_output = self.attention(q, k, v, attn_mask)
        attn_output = self._combine(attn_output, batch_size)
        attn_output = self.dropout(attn_output)
        return self.out_proj(attn_output)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.norm_first = norm_first
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            activation=activation or nn.GELU(),
            dropout=dropout,
        )
        self.mlp_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.norm_first:
            x = x + self.attn_dropout(self.self_attn(self.norm1(x), mask=mask))
            x = x + self.mlp_dropout(self.mlp(self.norm2(x)))
        else:
            attn_out = self.self_attn(x, mask=mask)
            x = self.norm1(x + self.attn_dropout(attn_out))
            x = self.norm2(x + self.mlp_dropout(self.mlp(x)))
        return x
