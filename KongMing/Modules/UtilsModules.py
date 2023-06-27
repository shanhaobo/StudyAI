from torch import nn

from einops.layers.torch import Rearrange

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

class DownsampleModule(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()

        self.Blocks = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(inInputDim * 4, inInputDim, 1),
        )

    def forward(self, inData):
        return self.Blocks(inData)


class UpsampleModule(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()

        self.Blocks = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(inInputDim, inInputDim, 3, padding=1),
        )

    def forward(self, inData):
        return self.Blocks(inData)

