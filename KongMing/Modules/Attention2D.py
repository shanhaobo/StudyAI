import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange

class IMultiHeadAttention2D(nn.Module):
    def __init__(self, inChannel, inNumHeads, inHeadEmbedDim):
        super(IMultiHeadAttention2D, self).__init__()

        self._inChannel      = inChannel
        self._NumHeads       = inNumHeads
        self._HeadEmbedDim   = inHeadEmbedDim
        self._HiddenEmbedDim = inHeadEmbedDim * inNumHeads
        self._ScaledFctr     = 1.0 / math.sqrt(self._HeadEmbedDim)
        self._AttLayer       = nn.Conv2d(in_channels = self._inChannel, out_channels = self._HiddenEmbedDim * 3,  kernel_size = 1, bias=False)
        self._AttOutLayer    = nn.Conv2d(in_channels = self._HiddenEmbedDim,  out_channels = inChannel,          kernel_size = 1)
    
    def _ToQKV(self, inX):
        # inX = (b, c_in = inChannel x, y)  ->  QKV = (b, c_out = self._EmbedDim * 3, x_out, y_out) 
        QKV = self._AttLayer(inX)
        # k = 3 because (q k v)
        # (k h d) means EmbedDim = c_out = self._EmbedDim * 3 也就是隐藏层, Conv2d输出的3倍隐藏层维度
        # (x y) means combie data 将数据维度合并
        Q, K, V = rearrange(QKV, "b (k h d) x y -> b k h d (x y)", k=3, h=self._NumHeads).chunk(3, dim=1)
        Q = Q.squeeze(1)
        K = K.squeeze(1)
        V = V.squeeze(1)

        return Q, K, V
    
    def CalcAtt(self, inX):
        _, _, h, w = inX.size()

        Q, K, V = self._ToQKV(inX)

        # d means embedding dim
        # i j means data dim
        # 其实就是将Q,K的数据组合到一起
        AttScaled = torch.einsum("b h d i, b h d j -> b h i j", Q, K)  * self._ScaledFctr

        # softmax
        # 上面的计算使得, 倒数第一个维度与倒数第二个维度都是数据的维度
        # 然后再倒数第一个维度上进行softmax
        # 这样数据与softmax就一一对应了
        Att = AttScaled - AttScaled.amax(dim=-1, keepdim=True).detach()
        Att = Att.softmax(dim=-1)

        # 
        Out = torch.einsum("b h i j, b h d j -> b h i d", Att, V)
        Out = rearrange(Out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        
        return self._AttOutLayer(Out)
    
    def CalcLinearAtt(self, inX):
        _, _, h, w = inX.size()

        Q, K, V = self._ToQKV(inX)

        Q = Q.softmax(dim=-2)
        K = K.softmax(dim=-1)
        
        Ctxt = torch.einsum("b h i j, b h d j -> b h i d", K, V)
        # d means embedding dim
        # i j means data dim
        # 其实就是将Q,K的数据组合到一起
        AttScaled = torch.einsum("b h d i, b h d j -> b h i j", Ctxt, Q)  * self._ScaledFctr
        Out = rearrange(AttScaled, "b h d (x y) -> b (h d) x y", x=h, y=w)
        
        return self._AttOutLayer(Out)
    

class MultiHeadAttention2D(IMultiHeadAttention2D):
    def __init__(self, inDim, inNumHeads, inHeadEmbedDim):
        super(MultiHeadAttention2D, self).__init__(inDim, inNumHeads, inHeadEmbedDim)
        
    def forward(self, inX):
        return self.CalcAtt(inX)

class MultiHeadLinearAttn2D(IMultiHeadAttention2D):
    def __init__(self, inDim, inNumHeads, inHeadEmbedDim):
        super(MultiHeadLinearAttn2D, self).__init__(inDim, inNumHeads, inHeadEmbedDim)
        self.GN = nn.GroupNorm(1, inDim)

    def forward(self, inX):
        return self.GN(self.CalcLinearAtt(inX))
