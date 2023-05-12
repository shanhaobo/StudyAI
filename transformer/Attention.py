import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class IMultiHeadAttention(nn.Module):
    def __init__(self, inEmbedDim, inNumHeads, inPosEmbedDim):
        super(IMultiHeadAttention, self).__init__()
        assert inEmbedDim % inNumHeads == 0

        self._NumHeads       = inNumHeads
        self._EmbedDim       = inEmbedDim
        self._HeadDim        = inEmbedDim // inNumHeads
        self._ScaledFctr     = 1.0 / math.sqrt(self._HeadDim)
        self._AttLayer       = nn.Linear(self._EmbedDim, self._EmbedDim * 3)
        self._AttOutLayer    = nn.Linear(self._EmbedDim, self._EmbedDim)
    
    def _QKV(self, inX):
        nBatchNum, nMaxSeqLen, nEmbedDim = inX.size() # [nBatchNum, nMaxSeqLen, nEmbedDim]

        # [BatchNum, MaxSeqLen, EmbedDim]->[BatchNum, NumHead, nMaxSeqLen, HeadDim]
        #      (B, S, E)->(B, S, 3 * E)->(B, S, E)->(B, S, H, HE)->(B, H, S, HE)
        # input ->   __Attlayer  ->  chunk   ->  view     ->  transpose
        Q, K, V = self._AttLayer(inX).chunk(3, dim=2)
        # 在transpose之前view, view要求contiguous()
        # 所以在transpose之前view保证了contiguous()
        Q = Q.view(nBatchNum, nMaxSeqLen, self._NumHeads, self._HeadDim).transpose(1, 2)
        K = K.view(nBatchNum, nMaxSeqLen, self._NumHeads, self._HeadDim).transpose(1, 2)
        V = V.view(nBatchNum, nMaxSeqLen, self._NumHeads, self._HeadDim).transpose(1, 2)

        """
        因为transpose在前, 所以只能使用reshape, 不能使用view
        Q, K, V = self.__AttLayer(inX).transpose(1, 2).chunk(3, dim=1)
        Q = Q.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        K = K.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        V = V.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        """

        return Q, K, V, nBatchNum, nMaxSeqLen, nEmbedDim

    def _CalcAtt(self, inX):
        nBatchNum, nMaxSeqLen, nEmbedDim = inX.size()

        # [BatchNum, MaxSeqLen, EmbedDim]->[BatchNum, NumHead, nMaxSeqLen, HeadDim]
        #      (B, S, E)->(B, S, 3 * E)->(B, S, E)->(B, S, H, HE)->(B, H, S, HE)
        # input ->   __Attlayer  ->  chunk   ->  view     ->  transpose
        Q, K, V = self._AttLayer(inX).chunk(3, dim=2)
        # 在transpose之前view, view要求contiguous()
        # 所以在transpose之前view保证了contiguous()
        Q = Q.view(nBatchNum, nMaxSeqLen, self._NumHeads, self._HeadDim).transpose(1, 2)
        K = K.view(nBatchNum, nMaxSeqLen, self._NumHeads, self._HeadDim).transpose(1, 2)
        V = V.view(nBatchNum, nMaxSeqLen, self._NumHeads, self._HeadDim).transpose(1, 2)

        """
        因为transpose在前, 所以只能使用reshape, 不能使用view
        Q, K, V = self.__AttLayer(inX).transpose(1, 2).chunk(3, dim=1)
        Q = Q.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        K = K.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        V = V.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        """
        # Scaled dot product attention
        Att = torch.matmul(Q, K.transpose(-1, -2)) * self._ScaledFctr
        
        Att = F.softmax(Att, dim=-1)

        Out = torch.matmul(Att, V)
        # 这里有两种写法,一种contiguous然后view, 另一种直接reshape(因为,它有额外操作,还是第一种更快)
        #Out = Out.transpose(1, 2).reshape(nBatchNum, nMaxSeqLen, nEmbedDim)
        Out = Out.transpose(1, 2).contiguous().view(nBatchNum, nMaxSeqLen, nEmbedDim)
 
        '''print(Out.size()) torch.Size([1, 3, 16])'''

        return self._AttOutLayer(Out)

    def _CalcAttWithFunctor(self, inX, AttFunc):
        nBatchNum, nMaxSeqLen, nEmbedDim = inX.size()

        # [BatchNum, MaxSeqLen, EmbedDim]->[BatchNum, NumHead, nMaxSeqLen, HeadDim]
        #      (B, S, E)->(B, S, 3 * E)->(B, S, E)->(B, S, H, HE)->(B, H, S, HE)
        # input ->   __Attlayer  ->  chunk   ->  view     ->  transpose
        Q, K, V = self._AttLayer(inX).chunk(3, dim=2)
        # 在transpose之前view, view要求contiguous()
        # 所以在transpose之前view保证了contiguous()
        Q = Q.view(nBatchNum, nMaxSeqLen, self._NumHeads, self._HeadDim).transpose(1, 2)
        K = K.view(nBatchNum, nMaxSeqLen, self._NumHeads, self._HeadDim).transpose(1, 2)
        V = V.view(nBatchNum, nMaxSeqLen, self._NumHeads, self._HeadDim).transpose(1, 2)

        """
        因为transpose在前, 所以只能使用reshape, 不能使用view
        Q, K, V = self.__AttLayer(inX).transpose(1, 2).chunk(3, dim=1)
        Q = Q.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        K = K.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        V = V.reshape(nBatchNum, self.__NumHeads, nMaxSeqLen, self.__HeadDim)
        """
        # Scaled dot product attention
        Att = torch.matmul(Q, K.transpose(-1, -2)) * self._ScaledFctr
        
        Att = AttFunc(Att)

        Att = F.softmax(Att, dim=-1)
        """
        print(Att):
        tensor([[
         [[1.0000, 0.0000, 0.0000],
          [0.4993, 0.5007, 0.0000],
          [0.3329, 0.3337, 0.3334]],

         [[1.0000, 0.0000, 0.0000],
          [0.4997, 0.5003, 0.0000],
          [0.3337, 0.3330, 0.3333]],

         [[1.0000, 0.0000, 0.0000],
          [0.5000, 0.5000, 0.0000],
          [0.3328, 0.3335, 0.3337]],

         [[1.0000, 0.0000, 0.0000],
          [0.5006, 0.4994, 0.0000],
          [0.3335, 0.3331, 0.3334]],

         [[1.0000, 0.0000, 0.0000],
          [0.5000, 0.5000, 0.0000],
          [0.3337, 0.3347, 0.3316]]]], grad_fn=<SoftmaxBackward0>
        """
        Out = torch.matmul(Att, V)
        
        '''
        print("Att:", Att.size(), "V:", V.size(), "Out:", Out.size())
        Att: torch.Size([1, 4, 3, 3]) V: torch.Size([1, 4, 3, 4]) Out: torch.Size([1, 4, 3, 4])
        '''

        # 这里有两种写法,一种contiguous然后view, 另一种直接reshape(因为,它有额外操作,还是第一种更快)
        # Out = Out.transpose(1, 2).reshape(nBatchNum, nMaxSeqLen, nEmbedDim)
        Out = Out.transpose(1, 2).contiguous().view(nBatchNum, nMaxSeqLen, nEmbedDim)
 
        '''print(Out.size()) torch.Size([1, 3, 16])'''

        return self._AttOutLayer(Out)

class MultiHeadAttention(IMultiHeadAttention):
    def __init__(self, inEmbedDim, inNumHeads, inPosEmbedDim):

        super(MultiHeadAttention, self).__init__(inEmbedDim, inNumHeads, inPosEmbedDim)
        
    def forward(self, inX):
        
        return super(MultiHeadAttention, self)._CalcAtt(inX)

class CausalSelfAttention(IMultiHeadAttention):
    def __init__(self, inEmbedDim, inNumHeads, inPosEmbedDim):
        super(CausalSelfAttention, self).__init__(inEmbedDim, inNumHeads, inPosEmbedDim)

        '''
        #print(self.__CausalMask)
        tensor([[[
          [False,  True,  True],
          [False, False,  True],
          [False, False, False]]]])
        '''
        self.__CausalMask     = torch.tril(torch.ones(1, 1, inPosEmbedDim, inPosEmbedDim)) == 0

    def __Mask(self, Att):
        # mask
        '''
        print(Att)
        tensor([[
         [[ 0.0073,    -inf,    -inf],
          [ 0.0017,  0.0016,    -inf],
          [ 0.0044, -0.0008, -0.0009]],

         [[-0.0035,    -inf,    -inf],
          [-0.0006, -0.0004,    -inf],
          [-0.0040, -0.0017, -0.0012]],

         [[-0.0022,    -inf,    -inf],
          [ 0.0029,  0.0003,    -inf],
          [ 0.0040,  0.0028,  0.0038]],

         [[ 0.0066,    -inf,    -inf],
          [ 0.0029,  0.0030,    -inf],
          [ 0.0053,  0.0083, -0.0008]]]], grad_fn=<MaskedFillBackward0>
        '''
        return Att.masked_fill(self.__CausalMask, float('-inf'))

    def forward(self, inX):
        
        return super(CausalSelfAttention, self)._CalcAttWithFunctor(inX, self.__Mask)
