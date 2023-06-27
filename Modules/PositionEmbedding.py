import torch
from torch import nn
import math

"""
====================================================================================
"""
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, inDim):
        super().__init__()
        self.Dim = inDim

    def forward(self, inTimesteps):
        Device = inTimesteps.device
        HalfDim = self.Dim // 2
        Embeddings = math.log(10000) / (HalfDim - 1)
        Embeddings = torch.exp(torch.arange(HalfDim, device=Device) * -Embeddings)
        Embeddings = inTimesteps[:, None] * Embeddings[None, :]
        Embeddings = torch.cat((Embeddings.sin(), Embeddings.cos()), dim=-1)
        return Embeddings
