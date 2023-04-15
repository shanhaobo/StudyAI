import torch
import torch.nn as nn

from .CausalSelfAttention import CausalSelfAttention

class TransformerBlockMLP(nn.Module):
    def __init__(self, inEmbedDim, inEnableBias):
        super().__init__()
        self.FCLayer    = nn.Linear(inEmbedDim, 4 * inEmbedDim, bias=inEnableBias)
        self.Activate   = nn.GELU()
        self.OutLayer   = nn.Linear(4 * inEmbedDim, inEmbedDim, bias=inEnableBias)

    def forward(self, inX):
        # (EmbedDim, 4 * EmbedDim) -> (4 * EmbedDim, EmbedDim)
        #                   FCLayer->OutLayer
        inX = self.Activate(self.FCLayer(inX))
        inX = self.OutLayer(inX)
        return inX

class TransformerBlock(nn.Module):
    def __init__(self, inEmbedDim, inHeadNum, inPosEmbedDim, inEnableBias, inEpsNorm):
        super().__init__()
        #LayerNorm 防止梯度爆炸与消失,从而加速训练过程并提高模型性能
        self.AttenLayerNorm = nn.LayerNorm(inEmbedDim, inEpsNorm)
        #自注意力机制允许模型根据输入序列中各个单词之间的关系来计算每个单词的表示
        self.AttenLayer     = CausalSelfAttention(inEmbedDim, inHeadNum, inPosEmbedDim)
        self.MLPLayerNorm   = nn.LayerNorm(inEmbedDim, inEpsNorm)
        self.MLPLayer       = TransformerBlockMLP(inEmbedDim, inEnableBias)

    def forward(self, inX):
        # 自注意力机制生成新单词,与输入(inX)相加形成了ResNet
        ResNetAtt = inX + self.AttenLayer(self.AttenLayerNorm(inX))
        # MLP 层的目的是对从自注意力层传递过来的信息进行更深入的非线性处理。
        # 这有助于提取更高级别的特征，从而提高模型的表示能力
        ResNetMLP = inX + self.MLPLayer(self.MLPLayerNorm(ResNetAtt))

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
        self.WordTokenEmbed = nn.Embedding(VocabSize, TokenEmbedDim)
        self.WordPosEmbed   = nn.Embedding(PosEmbedDim, TokenEmbedDim)
        self.Blocks         = nn.ModuleList([
                                TransformerBlock(
                                    TokenEmbedDim,
                                    HeadNum,
                                    PosEmbedDim,
                                    EnableBias,
                                    EpsNorm
                                ) for _ in range(BlockNum)
                             ])
        self.LayerNorm      = nn.LayerNorm(TokenEmbedDim)

    def forward(self, inToken, inPos):
        
        #Token : (nBatchNum, nTokenLen) -> (nBatchNum, nTokenLen, nEmbedDim)
        TokenEmbed = self.WordTokenEmbed(inToken)
        #Pos : (1, nTokenLen) -> (1, nTokenLen, nEmbedDim)
        PosEmbed = self.WordPosEmbed(inPos)

        TPEmbed = TokenEmbed + PosEmbed
        for block in self.Blocks:
            TPEmbed = block(TPEmbed)

        return self.LayerNorm(TPEmbed)

