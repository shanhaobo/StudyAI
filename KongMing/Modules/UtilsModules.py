from torch import nn

from einops.layers.torch import Rearrange

# Residual Network
class ResNet(nn.Module):
    def __init__(self, inFunc):
        super().__init__()
        self.Func = inFunc

    def forward(self, inData, inArgs, inKVArgs):
        return self.Func(inData, inArgs, inKVArgs) + inData

# 
class PreNorm(nn.Module):
    def __init__(self, inDim, inFunc):
        super().__init__()
        self.Func = inFunc
        self.Norm = nn.GroupNorm(1, inDim)

    def forward(self, inData):
        return self.Func(self.Norm(inData))

class DoubleLinearModule(nn.Module):
    def __init__(self, inInputDim, inOutputDim, inMidScale = 4) -> None:
        super().__init__()
        MidDim = inOutputDim * inMidScale
        self.Blocks = nn.Sequential(
            nn.Linear(inInputDim, MidDim),
            nn.GELU(),
            nn.Linear(MidDim, inOutputDim)
        )

    def forward(self, inData):
        return self.Blocks(inData)

class DoubleLinearModuleTO4D(nn.Module):
    def __init__(self, inInputDim, inOutputDim, inMidScale = 4) -> None:
        super().__init__()
        MidDim = inOutputDim * inMidScale
        self.Blocks = nn.Sequential(
            nn.Linear(inInputDim, MidDim),
            nn.GELU(),
            nn.Linear(MidDim, inOutputDim),
            Rearrange("b c -> b c 1 1")
        )

    def forward(self, inData):
        return self.Blocks(inData)

class DownsampleModule2D(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()

        self.Blocks = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(inInputDim * 4, inInputDim, 1),
        )

    def forward(self, inData):
        return self.Blocks(inData)

class UpsampleModule2D(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()

        self.Blocks = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(inInputDim, inInputDim, 3, padding=1),
        )

    def forward(self, inData):
        return self.Blocks(inData)
