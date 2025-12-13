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
                 vocab_size,
                 embed_dim,
                 dropout):
        
        super().__init__()

        



    
