import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .UNet2DBase import UNet2DBase, UNet2DBaseWithExtData, UNet2DBasePLUSExtData
from .PositionEmbedding import SinusoidalPositionEmbeddings
from .CustomConv2D import WeightStandardizedConv2D
from .Attention2D import MultiHeadAttention2D

# UNet的一大层，包含了两层小的卷积
class DoubleConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim):
        super(DoubleConv, self).__init__()

        Mid = (inInputDim + inOutputDim) // 2
        self.Blocks = nn.Sequential(
            nn.Conv2d(inInputDim, Mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(Mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(Mid, inOutputDim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inOutputDim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, inData):
        Output = self.Blocks(inData)
        #print("DoubleConv:{}=>{}".format(inData.size(), Output.size()))
        return Output

# 定义输入进来的第一层
class InputConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim):
        super(InputConv, self).__init__()
        self.Blocks = DoubleConv(inInputDim, inOutputDim)

    def forward(self, inData):
        return self.Blocks(inData)

# 定义最终的输出
class OutputConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim):
        super(OutputConv, self).__init__()
        self.Blocks = DoubleConv(inInputDim, inOutputDim)

    def forward(self, inData):
        #print("OutputConv:{}".format(inData.size()))
        return self.Blocks(inData)

################################################################################
################################################################################

class UNet2D(UNet2DBase):
    def __init__(self, inDim, inEmbedDim, inEmbedLvlCntORList) -> None:
        super().__init__(
            inDim,
            inDim,
            inEmbedDim,
            inEmbedLvlCntORList,
            InputConv,
            DoubleConv,
            DoubleConv,
            DoubleConv,
            OutputConv
        )

################################################################################
################################################################################

class UNet2DPosEmbed(UNet2DBasePLUSExtData):
    def __init__(self, inDim, inEmbedDim, inEmbedLvlCntORList) -> None:
        super().__init__(
            inDim,
            inDim,
            inEmbedDim,
            inEmbedLvlCntORList,
            InputConv,
            DoubleConv,
            DoubleConv,
            DoubleConv,
            OutputConv,
            SinusoidalPositionEmbeddings
        )

################################################################################
################################################################################

# UNet的一大层，包含了两层小的卷积
class DoubleConvEmbed(DoubleConv):
    def __init__(self, inInputDim, inOutputDim):
        super(DoubleConvEmbed, self).__init__(inInputDim, inOutputDim)

    def forward(self, inData, inExtraData):
        X = inData * (inExtraData + 1) + inExtraData
        return self.Blocks(X)

class UNet2DPLUSPosEmbed(UNet2DBaseWithExtData):
    def __init__(self, inDim, inEmbedDim, inEmbedLvlCntORList) -> None:
        super().__init__(
            inDim,
            inDim,
            inEmbedDim,
            inEmbedLvlCntORList,
            InputConv,
            DoubleConvEmbed,
            DoubleConvEmbed,
            DoubleConvEmbed,
            OutputConv,
            SinusoidalPositionEmbeddings
        )

################################################################################
################################################################################
class GroupAttn(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()
        self.Norm = nn.GroupNorm(1, inInputDim)
        self.Attn = MultiHeadAttention2D(inInputDim, 4, inInputDim // 4)

    def forward(self, inData):
        return inData + self.Norm(self.Attn(inData))


# UNet的一大层，包含了两层小的卷积
class DoubleConvAttnPosEmbed(nn.Module):
    def __init__(self, inInputDim, inOutputDim):
        super(DoubleConvAttnPosEmbed, self).__init__()

        self.Blocks = nn.Sequential(
            WeightStandardizedConv2D(inInputDim, inOutputDim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, inOutputDim),
            nn.SiLU(),
            GroupAttn(inOutputDim),
            WeightStandardizedConv2D(inOutputDim, inOutputDim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, inOutputDim),
            nn.SiLU()
        )
        
    def forward(self, inData, inExtraData):
        X = inData * (inExtraData + 1) + inExtraData
        return self.Blocks(X)

class UNet2DAttnPosEmbed_RestNetBlock(nn.Module):
    def __init__(self, inInputDim, inOutputDim):
        super(UNet2DAttnPosEmbed_RestNetBlock, self).__init__()
        self.Block = DoubleConvAttnPosEmbed(inInputDim, inOutputDim)
        self.ResBlock = nn.Conv2d(inInputDim, inOutputDim, 1)
        
    def forward(self, inData, inExtraData):
        return self.Block(inData, inExtraData) + self.ResBlock(inData)

class UNet2DAttnPosEmbed(UNet2DBaseWithExtData):
    def __init__(self, inChannel, inEmbedDim, inEmbedLvlCntORList) -> None:
        super().__init__(
            inChannel,
            inChannel,
            inEmbedDim,
            inEmbedLvlCntORList,
            InputConv,
            UNet2DAttnPosEmbed_RestNetBlock,
            UNet2DAttnPosEmbed_RestNetBlock,
            UNet2DAttnPosEmbed_RestNetBlock,
            OutputConv,
            SinusoidalPositionEmbeddings
        )
