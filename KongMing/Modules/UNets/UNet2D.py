import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .UNet2DBase import UNet2DBase, UNet2DBaseWithExtData
from ..PositionEmbedding import SinusoidalPositionEmbedding
from ..CustomEnhancedModules import WeightStandardizedConv2D
from ..Attention2D import MultiHeadAttention2D

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
    def __init__(self, inColorChanNum, inEmbeddingDim, inEmbedLvlCntORList) -> None:
        super().__init__(
            inColorChanNum,
            inColorChanNum,
            inEmbeddingDim,
            inEmbedLvlCntORList,
            InputConv,
            DoubleConv,
            DoubleConv,
            DoubleConv,
            OutputConv
        )

################################################################################
################################################################################

# UNet的一大层，包含了两层小的卷积
class DoubleConvEmbed(DoubleConv):
    def __init__(self, inInputDim, inOutputDim):
        super(DoubleConvEmbed, self).__init__(inInputDim, inOutputDim)

    def forward(self, inData, inExtraData):
        #X = inData * (inExtraData + 1) + inExtraData
        return self.Blocks(inData)

class UNet2DPLUSPosEmbed(UNet2DBaseWithExtData):
    def __init__(self, inDim, inEmbeddingDim, inEmbedLvlCntORList, inExtDataDim = None) -> None:
        super().__init__(
            inDim,
            inDim,
            inEmbeddingDim,
            inEmbedLvlCntORList,
            InputConv,
            DoubleConvEmbed,
            DoubleConvEmbed,
            DoubleConvEmbed,
            OutputConv,
            SinusoidalPositionEmbedding,
            inExtDataDim
        )

################################################################################
################################################################################
class ResGroupAttn(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()
        self.Norm = nn.GroupNorm(1, inInputDim)
        self.Attn = MultiHeadAttention2D(inInputDim, 4, 32)

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
            ResGroupAttn(inOutputDim),
            WeightStandardizedConv2D(inOutputDim, inOutputDim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, inOutputDim),
            nn.SiLU()
        )
        
    def forward(self, inData, inExtraData):
        X = inData * (inExtraData + 1) + inExtraData
        return self.Blocks(X)

class UNet2DPosEmbed_DoubleAttnResNetBlock(nn.Module):
    def __init__(self, inInputDim, inOutputDim):
        super(UNet2DPosEmbed_DoubleAttnResNetBlock, self).__init__()
        self.Block = DoubleConvAttnPosEmbed(inInputDim, inOutputDim)
        self.ResBlock = nn.Conv2d(inInputDim, inOutputDim, 1)
        
    def forward(self, inData, inExtraData):
        return self.Block(inData, inExtraData) + self.ResBlock(inData)

class UNet2DPosEmbed_DoubleAttn(UNet2DBaseWithExtData):
    def __init__(self, inColorChanNum, inEmbeddingDim, inEmbedLvlCntORList, inExtDataDim = None) -> None:
        super().__init__(
            inColorChanNum,
            inColorChanNum,
            inEmbeddingDim,
            inEmbedLvlCntORList,
            InputConv,
            UNet2DPosEmbed_DoubleAttnResNetBlock,
            UNet2DPosEmbed_DoubleAttnResNetBlock,
            UNet2DPosEmbed_DoubleAttnResNetBlock,
            OutputConv,
            SinusoidalPositionEmbedding,
            inExtDataDim
        )

################################################################################
################################################################################

# UNet的一大层，包含了两层小的卷积
class TripleConvAttnPosEmbed(nn.Module):
    def __init__(self, inInputDim, inOutputDim, inScale = 2):
        super().__init__()

        self.InConv = WeightStandardizedConv2D(inInputDim, inInputDim, kernel_size=7, padding=3, groups=inInputDim)
        self.Blocks = nn.Sequential(
            nn.GroupNorm(1, inInputDim),
            WeightStandardizedConv2D(inInputDim, inOutputDim * inScale, kernel_size=3, padding=1),
            nn.GroupNorm(1, inOutputDim * inScale),
            nn.GELU(),
            WeightStandardizedConv2D(inOutputDim * inScale, inOutputDim, kernel_size=3, padding=1),
        )
        
    def forward(self, inData, inExtraData):
        X = self.InConv(inData)
        return self.Blocks(X)

class UNet2DPosEmbed_TripleAttnResNetBlock(nn.Module):
    def __init__(self, inInputDim, inOutputDim):
        super().__init__()
        self.Block = TripleConvAttnPosEmbed(inInputDim, inOutputDim)
        self.ResBlock = nn.Conv2d(inInputDim, inOutputDim, kernel_size=3, padding=1)
        
    def forward(self, inData, inExtraData):
        return self.Block(inData, inExtraData) + self.ResBlock(inData)

class UNet2DPosEmbed_TripleAttn(UNet2DBaseWithExtData):
    def __init__(self, inColorChanNum, inEmbeddingDim, inEmbedLvlCntORList, inExtDataDim = None) -> None:
        super().__init__(
            inColorChanNum,
            inColorChanNum,
            inEmbeddingDim,
            inEmbedLvlCntORList,
            InputConv,
            UNet2DPosEmbed_TripleAttnResNetBlock,
            UNet2DPosEmbed_TripleAttnResNetBlock,
            UNet2DPosEmbed_TripleAttnResNetBlock,
            OutputConv,
            SinusoidalPositionEmbedding,
            inExtDataDim
        )

#########################################################################

from .UNet2DBase import UNet2DBaseWithExtData
from ..PositionEmbedding import SinusoidalPositionEmbedding
from ..UtilsModules import DoubleLinearModuleTO4D, ResNet, PreNorm
from ..Attention2D import MultiHeadAttention2D, MultiHeadLinearAttn2D

class CU2_ConvNeXtBlock(nn.Module):
    def __init__(self, dim, dim_out, *, mult=2):
        super().__init__()

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.ds_conv(x)

        h = self.net(h)
        return h + self.res_conv(x)

class CU2_InitConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim) -> None:
        super().__init__()

        self.Blocks = nn.Conv2d(inInputDim, inOutputDim, 7, padding=3)

    def forward(self, inData):
        return self.Blocks(inData)
    
class CU2_FinalConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim) -> None:
        super().__init__()

        self.Blocks =  nn.Sequential(
            CU2_ConvNeXtBlock(inInputDim, inInputDim),
            nn.Conv2d(inInputDim, inOutputDim, 1)
        )

    def forward(self, inData):
        return self.Blocks(inData)
    
class CU2_SampleConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim) -> None:
        super().__init__()
    
        self.b1 = CU2_ConvNeXtBlock(inInputDim, inOutputDim)
        self.eb1 = DoubleLinearModuleTO4D(inInputDim, inOutputDim)
        self.b2 = CU2_ConvNeXtBlock(inOutputDim, inOutputDim)
        self.eb2 = DoubleLinearModuleTO4D(inInputDim, inOutputDim)
        self.b3 = ResNet(PreNorm(inOutputDim, MultiHeadLinearAttn2D(inOutputDim, 4, 32)))

    def forward(self, inData, inExtData):
        x = self.b1(inData)
        e = self.eb1(inExtData)
        x = self.b2(x + e)
        e = self.eb2(inExtData)
        return self.b3(x + e)
    
class CU2_MidSampleConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim) -> None:
        super().__init__()
    
        self.b1 = CU2_ConvNeXtBlock(inInputDim, inOutputDim)
        self.eb1 = DoubleLinearModuleTO4D(inInputDim, inOutputDim)
        self.b2 = ResNet(PreNorm(inOutputDim, MultiHeadAttention2D(inOutputDim, 4, 32)))
        self.eb2 = DoubleLinearModuleTO4D(inOutputDim, inOutputDim)
        self.b3 = CU2_ConvNeXtBlock(inOutputDim, inOutputDim)

    def forward(self, inData, inExtData):
        x = self.b1(inData)
        e = self.eb1(inExtData)
        x = self.b2(x + e)
        e = self.eb2(inExtData)
        x = self.b3(x + e)
        return x
    
class UNet2D_ConvNeXt(UNet2DBaseWithExtData) :
    def __init__(self, inColorChanNum, inEmbeddingDim, inEmbedLvlCntORList, inExtDataDim = None) -> None:
        super().__init__(
            inColorChanNum,
            inColorChanNum,
            inEmbeddingDim,
            inEmbedLvlCntORList,
            CU2_InitConv,
            CU2_SampleConv,
            CU2_MidSampleConv,
            CU2_SampleConv,
            CU2_FinalConv,
            SinusoidalPositionEmbedding,
            inExtDataDim
        )
