from torch import nn

from einops.layers.torch import Rearrange

class Avg_2(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()

        self.Blocks = nn.AvgPool2d(2, stride=2)

    def forward(self, inData):
        return self.Blocks(inData)

class Max_2(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()

        self.Blocks = nn.MaxPool2d(2, stride=2)

    def forward(self, inData):
        return self.Blocks(inData)

class PixelShuffle_2(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()

        self.Blocks = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            # stride = 1 : kernel_size = 2 * padding + 1  -> 保持不变
            nn.Conv2d(inInputDim * 4, inInputDim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, inData):
        return self.Blocks(inData)

class Conv_2(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()
        
        # stride = 2 : kernel_size = 2 * padding + 2 -> 缩小至二分之一
        self.Blocks = nn.Conv2d(inInputDim, inInputDim, kernel_size=2, stride=2, padding=0)

    def forward(self, inData):
        return self.Blocks(inData)
