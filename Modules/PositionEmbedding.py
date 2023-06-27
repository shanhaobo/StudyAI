import torch
from torch import nn
import math

"""
====================================================================================
"""
## 为什么要加上sin 与 cos 都是为了将离散的数据与连续的数据关联起来
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, inDim):
        super().__init__()
        self.Dim = inDim

    def forward(self, inTimesteps):
        Device = inTimesteps.device
        HalfDim = self.Dim // 2
        Embedding = math.log(10000) / (HalfDim - 1)
        Embedding = torch.exp(torch.arange(HalfDim, device=Device) * -Embedding)
        Embedding = inTimesteps[:, None] * Embedding[None, :]
        Embedding = torch.cat((Embedding.sin(), Embedding.cos()), dim=-1)
        return Embedding
