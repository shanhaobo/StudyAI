from einops import rearrange

import torch.nn as nn

from .UNet2DBase import UNet2DBaseWithExtData
from ..PositionEmbedding import SinusoidalPositionEmbedding
from ..UtilsModules import DoubleLinearModuleTO4D, DoubleLinearModule, ResNet, PreNorm
from ..Attention2D import MultiHeadAttention2D, MultiHeadLinearAttn2D

from ..CustomEnhancedModules import WeightStandardizedConv2D

class UNet2D_InitConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim) -> None:
        super().__init__()

        self.Blocks = nn.Conv2d(inInputDim, inOutputDim, 7, padding=3)

    def forward(self, inData):
        return self.Blocks(inData)
    
class CU2_ConvNeXtBlock(nn.Module):
    def __init__(self, dim, dim_out, mult=2):
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
        e = rearrange(inExtData, "b c -> b c 1 1")
        x = self.b1(inData + e)
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
        e = rearrange(inExtData, "b c -> b c 1 1")
        x = self.b1(inData + e)
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
            UNet2D_InitConv,
            CU2_SampleConv,
            CU2_MidSampleConv,
            CU2_SampleConv,
            CU2_FinalConv,
            SinusoidalPositionEmbedding,
            inExtDataDim
        )

########################################################################
########################################################################

class WeightStandardizedBlock_ExtData(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2D(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.eproj = WeightStandardizedConv2D(dim, dim_out, 3, padding=1)

    def forward(self, x, inExtData):
        x = self.proj(x)
        x = self.norm(x)
        e = self.eproj(inExtData)
        x = self.act(x + e)
        return x

class WeightStandardizedBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2D(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class WeightStandardizedResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()

        self.block1 = WeightStandardizedBlock_ExtData(dim, dim_out, groups=groups)
        self.block2 = WeightStandardizedBlock(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x, time_emb):
        
        time_emb = rearrange(time_emb, "b c -> b c 1 1")

        h = self.block1(x, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)

class WeightStandardizedResnetBlock_WOEXTDATA(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()

        self.block1 = WeightStandardizedBlock(dim, dim_out, groups=groups)
        self.block2 = WeightStandardizedBlock(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x):
        
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

class WSB_FinalConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim) -> None:
        super().__init__()

        self.Blocks =  nn.Sequential(
            WeightStandardizedResnetBlock_WOEXTDATA(inInputDim, inInputDim),
            nn.Conv2d(inInputDim, inOutputDim, 1)
        )

    def forward(self, inData):
        return self.Blocks(inData)
    
class WSR_SampleConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim) -> None:
        super().__init__()
    
        self.b1 = WeightStandardizedResnetBlock(inInputDim, inOutputDim)
        self.eb1 = DoubleLinearModule(inInputDim, inOutputDim)
        self.b2 = WeightStandardizedResnetBlock(inOutputDim, inOutputDim)
        self.eb2 = DoubleLinearModuleTO4D(inInputDim, inOutputDim)
        self.b3 = ResNet(PreNorm(inOutputDim, MultiHeadLinearAttn2D(inOutputDim, 4, 32)))

    def forward(self, inData, inExtData):
        x = self.b1(inData, inExtData)
        e = self.eb1(inExtData)
        x = self.b2(x, e)
        e = self.eb2(inExtData)
        return self.b3(x + e)
    
class WSR_MidSampleConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim) -> None:
        super().__init__()
    
        self.b1 = WeightStandardizedResnetBlock(inInputDim, inOutputDim)
        self.eb1 = DoubleLinearModuleTO4D(inInputDim, inOutputDim)
        self.b2 = ResNet(PreNorm(inOutputDim, MultiHeadAttention2D(inOutputDim, 4, 32)))
        self.eb2 = DoubleLinearModule(inOutputDim, inOutputDim)
        self.b3 = WeightStandardizedResnetBlock(inOutputDim, inOutputDim)

    def forward(self, inData, inExtData):
        x = self.b1(inData, inExtData)
        e = self.eb1(inExtData)
        x = self.b2(x + e)
        e = self.eb2(inExtData)
        return self.b3(x, e)
    
class UNet2D_WSR(UNet2DBaseWithExtData) :
    def __init__(self, inColorChanNum, inEmbeddingDim, inEmbedLvlCntORList, inExtDataDim = None) -> None:
        super().__init__(
            inColorChanNum,
            inColorChanNum,
            inEmbeddingDim,
            inEmbedLvlCntORList,
            UNet2D_InitConv,
            WSR_SampleConv,
            WSR_MidSampleConv,
            WSR_SampleConv,
            WSB_FinalConv,
            SinusoidalPositionEmbedding,
            inExtDataDim
        )
