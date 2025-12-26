from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .core import PositionalEncoding, TransformerEncoderLayer


class ClassifierModel(nn.Module):
    """Transformer backbone producing review representations."""
    ''' Function: __init__
        Description: Initialize transformer model with configuration.
        Args:
            config : Model configuration specifying architecture parameters.
        Returns:
            None
    '''
    def __init__(self, config) -> None:
        super().__init__()

############################## Here is the actual data preparation for the transformers ###################################

        self.config = config

        self.token_embedding: Optional[nn.Embedding] = None

        if config.vocab_size is not None:
            
            self.token_embedding = nn.Embedding(config.vocab_size,
                                                config.embed_dim,
                                                padding_idx=0)
            self.input_proj      = None
        else:
            self.input_proj = nn.Linear(config.input_dim, config.embed_dim)

        max_positions       = config.max_length + (1 if config.use_cls_token else 0)

        self.position       = PositionalEncoding(max_len    = max_positions,
                                                 embed_dim  = config.embed_dim,
                                                 dropout    = config.dropout,
                                                 method     = "trainable")

############################## From here we start the actual Tranformer block ###########################################
        
        self.encoder        = nn.ModuleList(TransformerEncoderLayer(embedDim          = config.embed_dim,
                                                                    numHeads          = config.num_heads,
                                                                    mlpRatio          = config.mlp_ratio,
                                                                    dropout           = config.dropout,
                                                                    attentionDropout  = config.attention_dropout)
                                            for _ in range(config.depth))

############################## Connects the classification head ###########################################
        
        self.norm = nn.LayerNorm(config.embed_dim)

        # Classifier head is attached with the backbone
        if config.cls_head_dim:
            self.head = nn.Sequential(nn.Linear(config.embed_dim, config.cls_head_dim),
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


    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialise module weights following transformer conventions.
        
        :param module: Description
        :type module: nn.Module
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


    def forward_features(self,
                         inputs : Tensor,
                         attention_mask: Optional[Tensor] = None,
                        ) -> Tensor:
        """
        Produce pooled latent features prior to the classification head.
        
        :param inputs: Description
        :type inputs: Tensor
        :param attention_mask: Description
        :type attention_mask: Optional[Tensor]
        :return: Description
        :rtype: Tensor
        """
        
    
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
