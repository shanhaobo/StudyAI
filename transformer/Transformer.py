import torch
import torch.nn as nn

from .CausalSelfAttention import CausalSelfAttention

class _BlockMLP(nn.Module):
    def __init__(self, inEmbedDim, inEnableBias):
        super().__init__()
        self.__FCLayer  = nn.Linear(inEmbedDim, 4 * inEmbedDim, bias=inEnableBias)
        self.__Activate = nn.GELU()
        self.__OutLayer = nn.Linear(4 * inEmbedDim, inEmbedDim, bias=inEnableBias)

    def forward(self, inX):
        # (EmbedDim, 4 * EmbedDim) -> (4 * EmbedDim, EmbedDim)
        #                   FCLayer->OutLayer
        inX = self.__Activate(self.__FCLayer(inX))
        inX = self.__OutLayer(inX)
        return inX

class _TransformerBlock(nn.Module):
    def __init__(self, inEmbedDim, inHeadNum, inPosEmbedDim, inEnableBias, inEpsNorm):
        super().__init__()
        #LayerNorm 防止梯度爆炸与消失,从而加速训练过程并提高模型性能
        self.__AttLayerNorm = nn.LayerNorm(inEmbedDim, inEpsNorm)
        #自注意力机制允许模型根据输入序列中各个单词之间的关系来计算每个单词的表示
        self.__AttLayer     = CausalSelfAttention(inEmbedDim, inHeadNum, inPosEmbedDim)
        self.__MLPLayerNorm = nn.LayerNorm(inEmbedDim, inEpsNorm)
        self.__MLPLayer     = _BlockMLP(inEmbedDim, inEnableBias)

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
        
        for block in self.__Blocks:
            TPEmbed = block(TPEmbed)

        return self.__LayerNorm(TPEmbed)
