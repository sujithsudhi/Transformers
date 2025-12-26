import math
import torch
from torch import Tensor, nn
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 embedDim : int,
                 numHeads : int,
                 dropout  : float = 0.0):
        """
        Docstring for __init__
        
        :param self: Description
        :param embedDim: Description
        :type embedDim: int
        :param numHeads: Description
        :type numHeads: int
        :param dropout: Description
        :type dropout: float
        """
        super().__init__()
        
        if embedDim % numHeads != 0:
            raise ValueError("Embedded dimension should be divisible by number of heads.") 

        self.numHeads  = numHeads
        self.embedDim  = embedDim
        self.headDim  = self.embedDim // self.numHeads

        self.dropout   = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.Wq = nn.Linear(self.embedDim, self.embedDim, bias=True)
        self.Wk = nn.Linear(self.embedDim, self.embedDim, bias=True)
        self.Wv = nn.Linear(self.embedDim, self.embedDim, bias=True)
        self.Wo = nn.Linear(self.embedDim, self.embedDim, bias=True)

    def scaledDotProduct(self, q, k, v, mask = None):
        """
        Here we calculate the scaled dot product or attension scores using
        q, k, v
        
        :param self: Description
        :param q: Description
        :param k: Description
        :param v: Description
        :param mask: Description
        """

        dk = q.size()[-1]
        attention =  torch.matmul(q, k.transpose(-2,-1))
        attention =  attention / math.sqrt(dk)

        if mask is not None:
            mask      = mask.to(dtype=torch.bool, device=attention.device)
            attention = attention.masked_fill(~mask, float('-inf')) # This helps to make the value to zero after the softmax application
        
        attentionProb = torch.softmax(attention, dim=-1)

        attentionProb = self.dropout(attentionProb)

        selfAttention = torch.matmul(attentionProb, v)
        
        return selfAttention

        
    def splitHeads(self, x : Tensor):
        """
        We split the linear projection of input tokens sequence
        for each head
        Output dimesion after split would be 
        [BatchSize * numHeads, seqLength, headDim]
        [1 * 8, 256, 512/8]
        
        :param x: input sequence
        """
        batchSize, seqLen, _ = x.shape

        return (x.view(batchSize, seqLen, self.numHeads, self.headDim)
                .transpose(1,2)
                .contiguous()
                .view(batchSize * self.numHeads, seqLen, self.headDim)) 


    def combineHeads(self,
                     x         : Tensor, 
                     batchSize : int):
        """
        Here we combines the self attenstions back
        
        :param self: Description
        :param x: Description
        """

        seqLen = x.size(1)
        return (x.view(batchSize, self.numHeads, seqLen, self.headDim)
                .transpose(1,2)
                .contiguous()
                .view(batchSize, seqLen, self.embedDim))
    

    def forward(self, 
                x    : Tensor, 
                mask : Optional[Tensor] = None):
        """
        This is the main part of the self attention block.
        passes the input to self linear projection, split heads,
        self attension(scaled dot product), compine heads, 
        dropout and output projection.
        
        :param self: Description
        :param Q: Description
        :param K: Description
        """
        batch_size = x.size(0)

        # Passes the linear projection to the split head
        Q = self.splitHeads(self.Wq(x))
        K = self.splitHeads(self.Wk(x))
        V = self.splitHeads(self.Wv(x))

        # After the splitting Q, K, V size would be [batchSize * numHeads, seqLength, embedDim]
        
        attn_mask = None
        if mask is not None:
            
            if mask.dtype != torch.bool:
                mask = mask > 0
            
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 3:
                pass
            else:
                raise ValueError("Unsupported attention mask rank.")
            
            if mask.size(0) == batch_size:
                mask = mask.repeat_interleave(self.numHeads, dim=0)
            elif mask.size(0) != batch_size * self.numHeads:
                raise ValueError("Attention mask batch dimension mismatch.")
            attn_mask = mask.to(Q.device)

        # Calcualating self attention
        attention = self.scaledDotProduct(q=Q,
                                          k=K,
                                          v=V,
                                          mask=attn_mask)
        
        # Combining the heads
        attention = self.combineHeads(attention, batch_size)
        attention = self.dropout(attention)

        # Output projection
        attention = self.Wo(attention)

        return attention
        

class ResidualBlock(nn.Module):
    def __init__(self, 
                 embedDim  : int,
                 module    : nn.Module,
                 dropout   : float = 0.0,
                 normFirst : bool  = True):
        """
        Docstring for __init__
        
        :param self: Description
        :param embedDim: Description
        :type embedDim: int
        :param module: Description
        :type module: nn.Module
        :param dropout: Description
        :type dropout: float
        :param normFirst: Description
        :type normFirst: bool
        """
        super().__init__()

        self.normFirst = normFirst
        self.module    = module
        self.dropout   = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        self.norm  = nn.LayerNorm(embedDim)

    def forward(self, 
                x : Tensor,
                *args, 
                **kwargs):
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description
        :type x: Tensor
        :param args: Description
        :param kwargs: Description
        """
        
        # Calculating residual
        if self.normFirst:  
            residue = x + self.dropout(self.module(self.norm(x),*args, **kwargs))
        else:
            residue = self.norm(x + self.dropout(self.module(x, *args, **kwargs)))

        return residue

class FeedForward(nn.Module):
    def __init__(self, 
                 inputDim   : int,
                 hiddenDim  : int,
                 outputDim  : int,
                 activation : nn.Module, 
                 dropout    : float = 0.0,
                 bias       : bool  = True):
        """
        Docstring for __init__
        
        :param self: Description
        :param inputDim: Description
        :type inputDim: int
        :param hiddenDim: Description
        :type hiddenDim: int
        :param outputDim: Description
        :type outputDim: int
        :param activation: Description
        :type activation: nn.Module
        :param dropout: Description
        :type dropout: float
        :param bias: Description
        :type bias: bool
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.fullyConnected1 = nn.Linear(inputDim, hiddenDim, bias=bias)
        self.fullyConnected2 = nn.Linear(hiddenDim, outputDim, bias=bias)

        self.activation      = activation or nn.GELU()

        # both input and ouptut dimension should be same
        outputDim            = inputDim or outputDim

    def forward(self, x : Tensor):
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description
        """
        x = self.fullyConnected1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fullyConnected2(x)

        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 embedDim         : int,
                 numHeads         : int,
                 mlpRatio         : int = 4,
                 activation       : Optional[nn.Module] = None,
                 attentionDropout : float = 0.0,
                 dropout          : float = 0.0,
                 normFirst        : bool  = True):
        """
        Docstring for __init__
        
        :param self: Description
        :param embedDim: Description
        :type embedDim: int
        :param numHeads: Description
        :type numHeads: int
        :param mlpRatio: Description
        :type mlpRatio: int
        :param activation: Description
        :type activation: Optional[nn.Module]
        :param attentionDropout: Description
        :type attentionDropout: float
        :param dropout: Description
        :type dropout: float
        :param normFirst: Description
        :type normFirst: bool
        """
        
        super().__init__()

        hiddenDim        = int(embedDim * mlpRatio)
        self.normFirst   = normFirst

        self.attention   = MultiHeadSelfAttention(embedDim = embedDim,
                                                  numHeads = numHeads,
                                                  dropout  = attentionDropout)
        
        self.residue1    = ResidualBlock(embedDim  = embedDim,
                                         dropout   = dropout, 
                                         module    = self.attention,
                                         normFirst = normFirst)


        self.ff          = FeedForward(inputDim   = embedDim,
                                       hiddenDim  = hiddenDim,
                                       outputDim  = embedDim,
                                       activation = activation or nn.GELU(),
                                       dropout    = dropout)
        
        self.residue2    = ResidualBlock(embedDim  = embedDim,
                                         dropout   = dropout, 
                                         module    = self.ff,
                                         normFirst = normFirst)
        
    def forward(self,
                x    : Tensor, 
                mask : Optional[Tensor] = None):
        """
        Docstring for forward
        
        :param self: Description
        :param x: Description
        """

        if self.normFirst:
            x = self.residue1(x,mask=mask) # Attenstion will be done inside the residual block
            
            x = self.residue2(x)
        else:
            x = self.residue1(x,mask=mask)

            x = self.residue2(x)
        
        return x




