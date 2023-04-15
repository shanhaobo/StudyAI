import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, inEmbedDim, inNumHeads, inPosEmbedDim):
        super().__init__()
        assert inEmbedDim % inNumHeads == 0

        self.NumHeads   = inNumHeads
        self.EmbedDim   = inEmbedDim
        self.HeadDim    = inEmbedDim // inNumHeads
        self.ScaledFctr = 1.0 / math.sqrt(self.HeadDim)
        self.AttenLayer = nn.Linear(self.EmbedDim, self.EmbedDim * 3)
        self.OutLayer   = nn.Linear(self.EmbedDim, self.EmbedDim)
        self.Mask       = torch.tril(torch.ones(1, 1, inPosEmbedDim, inPosEmbedDim)) == 0

    def forward(self, inX):
        nBatchNum, nMaxSeqLen, nEmbedDim = inX.size() # [nBatchNum, nMaxSeqLen, nEmbedDim]

        # [BatchNum, MaxSeqLen, EmbedDim]->[BatchNum, NumHead, nMaxSeqLen, HeadDim]
        #(B, S, E)->(B, S, 3 * E)->(B, S, E)->(B, S, H, HE)->(B, H, S, HE)
        Q, K, V = self.AttenLayer(inX).chunk(3, dim=2)
        # 在transpose之前view, view要求contiguous()
        # 所以在transpose之前view保证了contiguous()
        Q = Q.view(nBatchNum, nMaxSeqLen, self.NumHeads, self.HeadDim).transpose(1, 2)
        K = K.view(nBatchNum, nMaxSeqLen, self.NumHeads, self.HeadDim).transpose(1, 2)
        V = V.view(nBatchNum, nMaxSeqLen, self.NumHeads, self.HeadDim).transpose(1, 2)
        """
        因为transpose在前, 所以只能使用reshape, 不能使用view
        Q, K, V = self.AttenLayer(inX).transpose(1, 2).chunk(3, dim=1)
        Q = Q.reshape(nBatchNum, self.NumHeads, nMaxSeqLen, self.HeadDim)
        K = K.reshape(nBatchNum, self.NumHeads, nMaxSeqLen, self.HeadDim)
        V = V.reshape(nBatchNum, self.NumHeads, nMaxSeqLen, self.HeadDim)
        """

        # Scaled dot product attention
        Att = torch.matmul(Q, K.transpose(-1, -2)) * self.ScaledFctr

        Att = Att.masked_fill(self.Mask, float('-inf'))
        Att = F.softmax(Att, dim=-1)

        Out = torch.matmul(Att, V)
        # 这里有两种写法,一种contiguous然后view, 另一种直接reshape
        # Out = Out.transpose(1, 2).contiguous().view(nBatchNum, nMaxSeqLen, nEmbedDim)
        Out = Out.transpose(1, 2).reshape(nBatchNum, nMaxSeqLen, nEmbedDim)
 
        return self.OutLayer(Out)
