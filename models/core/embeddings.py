"""Embedding modules used by the transformers model."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """Lookup embedding layer with optional padding index."""

    ''' Function: __init__
        Description: Initialize token embedding layer with vocabulary size and embedding dimension.
        Args:
            vocab_size  : Size of the vocabulary.
            embed_dim   : Dimensionality of embedding vectors.
            padding_idx : Index for padding tokens (optional).
        Returns:
            None
    '''
    def __init__(self, 
                 vocab_size : int  = 256, 
                 embed_dim  : int  = 256, 
                 padding_idx = None):
        
        super().__init__()
        
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(num_embeddings = self.vocab_size,
                                      embedding_dim  = self.embed_dim,
                                      padding_idx    = self.padding_idx)
        
    def forward(self,tokens):
        """
        Docstring for forward
        
        :param self: Description
        """
        return self.embedding(tokens)
        

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with optional dropout."""

    ''' Function: __init__
        Description: Initialize sinusoidal positional encoding with dropout.
        Args:
            embed_dim : Dimensionality of embeddings.
            max_len   : Maximum sequence length supported.
            dropout   : Dropout probability applied after adding positional encoding.
        Returns:
            None
    '''

    def __init__(self, 
                 seq_len,
                 embed_dim,
                 dropout):
        
        super().__init__()

        self.embed_dim      = embed_dim
        self.seq_len        = seq_len
        self.dropout        = nn.Dropout(p=dropout)

        position            = torch.arange(0, seq_len).unsqueeze(1)
        
        div_term            = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / self.embed_dim))
        
        self.pe             = torch.zeros(seq_len, embed_dim)

        self.pe[:, 0::2]    = torch.sin(position * div_term)
        self.pe[:, 0::1]    = torch.cos(position * div_term)

        self.pe             = self.pe.unsqueeze(0)  # Shape: (1, seq_len, embed_dim)
        self.register_buffer('pe', self.pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input embeddings and apply dropout.

        :param x: Input tensor of shape (batch_size, seq_len, embed_dim).
        :return: Tensor of same shape as input with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

        


        



    
