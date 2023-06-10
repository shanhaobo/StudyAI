import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange

class IMultiHeadAttention2D(nn.Module):
    def __init__(self, inChannels, inNumHeads, inHeadEmbedDim):
        super(IMultiHeadAttention2D, self).__init__()

        self._InChannels     = inChannels
        self._NumHeads       = inNumHeads
        self._HeadEmbedDim   = inHeadEmbedDim
        self._EmbedDim       = inHeadEmbedDim * inNumHeads
        self._ScaledFctr     = 1.0 / math.sqrt(self._HeadEmbedDim)
        self._AttLayer       = nn.Conv2d(in_channels = self._InChannels,out_channels = self._EmbedDim * 3,  kernel_size = 1, bias=False)
        self._AttOutLayer    = nn.Conv2d(in_channels = self._EmbedDim,  out_channels = inChannels,          kernel_size = 1)
    
    def _ToQKV(self, inX):
        # inX = (b, c_in = inChannels x, y)  ->  QKV = (b, c_out = self._EmbedDim * 3, x_out, y_out) 
        QKV = self._AttLayer(inX)
        # k = 3 because (q k v)
        # (k h d) means EmbedDim = c_out = self._EmbedDim * 3
        Q, K, V = rearrange(QKV, "b (k h d) x y -> b k h d (x y)", k=3, h=self._NumHeads).chunk(3, dim=1)
        Q = Q.squeeze(1)
        K = K.squeeze(1)
        V = V.squeeze(1)

        return Q, K, V
    
    def _CalcAtt(self, inX):
        Q, K, V = self._ToQKV(inX)

        Att = torch.einsum("b h i s, b h j s -> b h i j", Q, K)  * self._ScaledFctr

        pass
