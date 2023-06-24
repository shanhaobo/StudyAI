import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .UNet2DBase import UNet2DBase, UNet2DBaseWithExtraData

from .PositionEmbedding import SinusoidalPositionEmbeddings

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
        self.Blocks = nn.Conv2d(inInputChannels, inOutputChannels, 1)

    def forward(self, inData):
        #print("OutputConv:{}".format(inData.size()))
        return self.Blocks(inData)

class UNet2D(UNet2DBase):
    def __init__(self, inChannels, inEmbedDims, inLevelCount) -> None:
        super().__init__(inChannels, inChannels, inEmbedDims, inLevelCount, InputConv, DoubleConv, DoubleConv, DoubleConv, OutputConv)

# UNet的一大层，包含了两层小的卷积
class DoubleConvEmbed(nn.Module):
    def __init__(self, inInputChannels, inOutputChannels, inExtraDataDims = None):
        super(DoubleConvEmbed, self).__init__()

        Mid = (inInputChannels + inOutputChannels) // 2
        self.Blocks = nn.Sequential(
            nn.Conv2d(inInputChannels, Mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(Mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(Mid, inOutputChannels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inOutputChannels),
            nn.ReLU(inplace=True)
        )

        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(inExtraDataDims, inInputChannels))
            if inExtraDataDims is not None
            else None
        )
        
    def forward(self, inData, inExtraData):
        h = self.mlp(inExtraData)

        h = rearrange(h, "b c -> b c 1 1")
        print("h:{}=>{}".format(inData.size(), h.size()))
        Output = self.Blocks(inData + h)
        #print("DoubleConv:{}=>{}".format(inData.size(), Output.size()))
        return Output

class UNet2DPosEmbed(UNet2DBaseWithExtraData):
    def __init__(self, inChannels, inEmbedDims, inLevelCount) -> None:
        super().__init__(inChannels, inChannels, inEmbedDims, inLevelCount, InputConv, DoubleConvEmbed, DoubleConvEmbed, DoubleConvEmbed, OutputConv, SinusoidalPositionEmbeddings)

