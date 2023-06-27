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
    def __init__(self, inInputChannels, inOutputChannels):
        super(DoubleConv, self).__init__()

        Mid = (inInputChannels + inOutputChannels) // 2
        self.Blocks = nn.Sequential(
            nn.Conv2d(inInputChannels, Mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(Mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(Mid, inOutputChannels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inOutputChannels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, inData):
        Output = self.Blocks(inData)
        #print("DoubleConv:{}=>{}".format(inData.size(), Output.size()))
        return Output

# 定义输入进来的第一层
class InputConv(nn.Module):
    def __init__(self, inInputChannels, inOutputChannels):
        super(InputConv, self).__init__()
        self.Blocks = DoubleConv(inInputChannels, inOutputChannels)

    def forward(self, inData):
        return self.Blocks(inData)

# 定义最终的输出
class OutputConv(nn.Module):
    def __init__(self, inInputChannels, inOutputChannels):
        super(OutputConv, self).__init__()
        self.Blocks = DoubleConv(inInputChannels, inOutputChannels)

    def forward(self, inData):
        #print("OutputConv:{}".format(inData.size()))
        return self.Blocks(inData)

################################################################################
################################################################################

class UNet2D(UNet2DBase):
    def __init__(self, inChannels, inEmbedDims, inLevelCount) -> None:
        super().__init__(
            inChannels,
            inChannels,
            inEmbedDims,
            inLevelCount,
            InputConv,
            DoubleConv,
            DoubleConv,
            DoubleConv,
            OutputConv
        )

################################################################################
################################################################################

class UNet2DPosEmbed(UNet2DBasePLUSExtData):
    def __init__(self, inChannels, inEmbedDims, inLevelCount) -> None:
        super().__init__(
            inChannels,
            inChannels,
            inEmbedDims,
            inLevelCount,
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
    def __init__(self, inInputChannels, inOutputChannels):
        super(DoubleConvEmbed, self).__init__(inInputChannels, inOutputChannels)

    def forward(self, inData, inExtraData):
        X = inData * (inExtraData + 1) + inExtraData
        return self.Blocks(X)

class UNet2DPLUSPosEmbed(UNet2DBaseWithExtData):
    def __init__(self, inChannels, inEmbedDims, inLevelCount) -> None:
        super().__init__(
            inChannels,
            inChannels,
            inEmbedDims,
            inLevelCount,
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
    def __init__(self, inInputChannels) -> None:
        super().__init__()
        self.Norm = nn.GroupNorm(1, inInputChannels)
        self.Attn = MultiHeadAttention2D(inInputChannels, 4, inInputChannels // 4)

    def forward(self, inData):
        return inData + self.Norm(self.Attn(inData))


# UNet的一大层，包含了两层小的卷积
class DoubleConvAttnPosEmbed(nn.Module):
    def __init__(self, inInputDims, inOutputDims):
        super(DoubleConvAttnPosEmbed, self).__init__()

        self.Blocks = nn.Sequential(
            WeightStandardizedConv2D(inInputDims, inOutputDims, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, inOutputDims),
            nn.SiLU(),
            GroupAttn(inOutputDims),
            WeightStandardizedConv2D(inOutputDims, inOutputDims, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, inOutputDims),
            nn.SiLU()
        )
        
    def forward(self, inData, inExtraData):
        X = inData * (inExtraData + 1) + inExtraData
        return self.Blocks(X)

class UNet2DAttnPosEmbed_RestNetBlock(nn.Module):
    def __init__(self, inInputChannels, inOutputChannels):
        super(UNet2DAttnPosEmbed_RestNetBlock, self).__init__()
        self.Block = DoubleConvAttnPosEmbed(inInputChannels, inOutputChannels)
        self.ResBlock = nn.Conv2d(inInputChannels, inOutputChannels, 1)
        
    def forward(self, inData, inExtraData):
        return self.Block(inData, inExtraData) + self.ResBlock(inData)

class UNet2DAttnPosEmbed(UNet2DBaseWithExtData):
    def __init__(self, inChannels, inEmbedDims, inLevelCount) -> None:
        super().__init__(
            inChannels,
            inChannels,
            inEmbedDims,
            inLevelCount,
            InputConv,
            UNet2DAttnPosEmbed_RestNetBlock,
            UNet2DAttnPosEmbed_RestNetBlock,
            UNet2DAttnPosEmbed_RestNetBlock,
            OutputConv,
            SinusoidalPositionEmbeddings
        )
