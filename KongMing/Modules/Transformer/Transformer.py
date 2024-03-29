import torch
import torch.nn as nn

from ..Attentions.Attention import CausalSelfAttention

class _BlockMLP(nn.Module):
    def __init__(self, inEmbeddingDim, inEnableBias):
        super().__init__()
        self.__FCLayer  = nn.Linear(inEmbeddingDim, 4 * inEmbeddingDim, bias=inEnableBias)
        self.__Activate = nn.GELU()
        self.__OutLayer = nn.Linear(4 * inEmbeddingDim, inEmbeddingDim, bias=inEnableBias)

    def forward(self, inX):
        # (EmbedDim, 4 * EmbedDim) -> (4 * EmbedDim, EmbedDim)
        #                   FCLayer->OutLayer
        inX = self.__Activate(self.__FCLayer(inX))
        inX = self.__OutLayer(inX)
        return inX

# 一次Attention
class _TransformerBlock(nn.Module):
    def __init__(self, inEmbeddingDim, inHeadNum, inPosEmbedDim, inEnableBias, inEpsNorm):
        super().__init__()
        #LayerNorm 防止梯度爆炸与消失,从而加速训练过程并提高模型性能
        self.__AttLayerNorm = nn.LayerNorm(inEmbeddingDim, inEpsNorm)
        #自注意力机制允许模型根据输入序列中各个单词之间的关系来计算每个单词的表示
        self.__AttLayer     = CausalSelfAttention(inHeadNum, inEmbeddingDim // inHeadNum, inPosEmbedDim)
        self.__MLPLayerNorm = nn.LayerNorm(inEmbeddingDim, inEpsNorm)
        self.__MLPLayer     = _BlockMLP(inEmbeddingDim, inEnableBias)

    def forward(self, inX):
        # 自注意力机制生成新单词,与输入(inX)相加形成了ResNet
        ResNetAtt = inX + self.__AttLayer(self.__AttLayerNorm(inX))
        # MLP 层的目的是对从自注意力层传递过来的信息进行更深入的非线性处理。
        # 这有助于提取更高级别的特征，从而提高模型的表示能力
        ResNetMLP = inX + self.__MLPLayer(self.__MLPLayerNorm(ResNetAtt))

        return ResNetMLP

class Transformer(nn.Module) :
    def __init__(
            self,
            VocabSize,
            TokenEmbedDim,
            PosEmbedDim,
            HeadNum,
            BlockNum,
            EpsNorm,
            EnableBias
        ) -> None:
        super().__init__()
        # Token Embedding
        self._WordTknEmbed  = nn.Embedding(VocabSize, TokenEmbedDim)
        # Pos Embedding
        self._WordPosEmbed  = nn.Embedding(PosEmbedDim, TokenEmbedDim)
        # Blocks
        self.__Blocks       = nn.ModuleList([
                                _TransformerBlock(
                                    TokenEmbedDim,
                                    HeadNum,
                                    PosEmbedDim,
                                    EnableBias,
                                    EpsNorm
                                ) for _ in range(BlockNum)
                             ])
        # 最后LayerNorm防止梯度失效
        self.__LayerNorm    = nn.LayerNorm(TokenEmbedDim)

    # Token Embedding + Pos Embedding -> Blocks -> LayerNorm
    def forward(self, inToken, inPos):
        
        #Token : (nBatchNum, nTokenLen) -> (nBatchNum, nTokenLen, nEmbedDim)
        TokenEmbed  = self._WordTknEmbed(inToken)
        #Pos : (1, nTokenLen) -> (1, nTokenLen, nEmbedDim)
        PosEmbed    = self._WordPosEmbed(inPos)
        # Total Embedding
        TPEmbed     = TokenEmbed + PosEmbed
        
        # 多次Attention, 每次Attention在前一次之上
        for block in self.__Blocks:
            TPEmbed = block(TPEmbed)

        return self.__LayerNorm(TPEmbed)
