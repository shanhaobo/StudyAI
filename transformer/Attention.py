import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, inEmbedDim, inNumHeads, inPosEmbedDim):
        super().__init__()
        assert inEmbedDim % inNumHeads == 0

        self.__NumHeads       = inNumHeads
        self.__EmbedDim       = inEmbedDim
        self.__HeadDim        = inEmbedDim // inNumHeads
        self.__ScaledFctr     = 1.0 / math.sqrt(self.__HeadDim)
        self.__AttLayer       = nn.Linear(self.__EmbedDim, self.__EmbedDim * 3)
        self.__AttOutLayer    = nn.Linear(self.__EmbedDim, self.__EmbedDim)
        self.__CausalMask     = torch.tril(torch.ones(1, 1, inPosEmbedDim, inPosEmbedDim)) == 0

    def forward(self, inX):
        nBatchNum, nMaxSeqLen, nEmbedDim = inX.size() # [nBatchNum, nMaxSeqLen, nEmbedDim]

        # [BatchNum, MaxSeqLen, EmbedDim]->[BatchNum, NumHead, nMaxSeqLen, HeadDim]
        #      (B, S, E)->(B, S, 3 * E)->(B, S, E)->(B, S, H, HE)->(B, H, S, HE)
        # input ->   __Attlayer  ->  chunk   ->  view     ->  transpose
        Q, K, V = self.__AttLayer(inX).chunk(3, dim=2)
        # 在transpose之前view, view要求contiguous()
        # 所以在transpose之前view保证了contiguous()
        Q = Q.view(nBatchNum, nMaxSeqLen, self.__NumHeads, self.__HeadDim).transpose(1, 2)
        K = K.view(nBatchNum, nMaxSeqLen, self.__NumHeads, self.__HeadDim).transpose(1, 2)
        V = V.view(nBatchNum, nMaxSeqLen, self.__NumHeads, self.__HeadDim).transpose(1, 2)
        """
        因为transpose在前, 所以只能使用reshape, 不能使用view
        Q, K, V = self.__AttLayer(inX).transpose(1, 2).chunk(3, dim=1)
        Q = Q.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        K = K.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        V = V.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        """

        # Scaled dot product attention
        Att = torch.matmul(Q, K.transpose(-1, -2)) * self.__ScaledFctr

        Att = Att.masked_fill(self.__CausalMask, float('-inf'))
        Att = F.softmax(Att, dim=-1)

        Out = torch.matmul(Att, V)
        # 这里有两种写法,一种contiguous然后view, 另一种直接reshape(因为,它有额外操作,还是第一种更快)
        #Out = Out.transpose(1, 2).reshape(nBatchNum, nMaxSeqLen, nEmbedDim)
        Out = Out.transpose(1, 2).contiguous().view(nBatchNum, nMaxSeqLen, nEmbedDim)
 
        return self.__AttOutLayer(Out)

class CausalSelfAttention(nn.Module):
    def __init__(self, inEmbedDim, inNumHeads, inPosEmbedDim):
        super().__init__()
        assert inEmbedDim % inNumHeads == 0

        self.__NumHeads       = inNumHeads
        self.__EmbedDim       = inEmbedDim
        self.__HeadDim        = inEmbedDim // inNumHeads
        self.__ScaledFctr     = 1.0 / math.sqrt(self.__HeadDim)
        self.__AttLayer       = nn.Linear(self.__EmbedDim, self.__EmbedDim * 3)
        self.__AttOutLayer    = nn.Linear(self.__EmbedDim, self.__EmbedDim)
        self.__CausalMask     = torch.tril(torch.ones(1, 1, inPosEmbedDim, inPosEmbedDim)) == 0

    def forward(self, inX):
        nBatchNum, nMaxSeqLen, nEmbedDim = inX.size() # [nBatchNum, nMaxSeqLen, nEmbedDim]

        # [BatchNum, MaxSeqLen, EmbedDim]->[BatchNum, NumHead, nMaxSeqLen, HeadDim]
        #      (B, S, E)->(B, S, 3 * E)->(B, S, E)->(B, S, H, HE)->(B, H, S, HE)
        # input ->   __Attlayer  ->  chunk   ->  view     ->  transpose
        Q, K, V = self.__AttLayer(inX).chunk(3, dim=2)
        # 在transpose之前view, view要求contiguous()
        # 所以在transpose之前view保证了contiguous()
        Q = Q.view(nBatchNum, nMaxSeqLen, self.__NumHeads, self.__HeadDim).transpose(1, 2)
        K = K.view(nBatchNum, nMaxSeqLen, self.__NumHeads, self.__HeadDim).transpose(1, 2)
        V = V.view(nBatchNum, nMaxSeqLen, self.__NumHeads, self.__HeadDim).transpose(1, 2)
        """
        因为transpose在前, 所以只能使用reshape, 不能使用view
        Q, K, V = self.__AttLayer(inX).transpose(1, 2).chunk(3, dim=1)
        Q = Q.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        K = K.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        V = V.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        """

        # Scaled dot product attention
        Att = torch.matmul(Q, K.transpose(-1, -2)) * self.__ScaledFctr

        Att = Att.masked_fill(self.__CausalMask, float('-inf'))
        Att = F.softmax(Att, dim=-1)

        Out = torch.matmul(Att, V)
        # 这里有两种写法,一种contiguous然后view, 另一种直接reshape(因为,它有额外操作,还是第一种更快)
        #Out = Out.transpose(1, 2).reshape(nBatchNum, nMaxSeqLen, nEmbedDim)
        Out = Out.transpose(1, 2).contiguous().view(nBatchNum, nMaxSeqLen, nEmbedDim)
 
        return self.__AttOutLayer(Out)
