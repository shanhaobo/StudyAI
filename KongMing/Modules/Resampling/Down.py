import torch
from torch import nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

from math import sqrt

########################################

class Avg_2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.Blocks = nn.AvgPool2d(2, stride=2)

    def forward(self, inData):
        return self.Blocks(inData)

########################################

class Max_2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.Blocks = nn.MaxPool2d(2, stride=2)

    def forward(self, inData):
        return self.Blocks(inData)

########################################

class PixelShuffle(nn.Module):
    def __init__(self, inInputDim, inMultiple : int) -> None:
        super().__init__()

        self.Blocks = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1 = inMultiple, p2 = inMultiple),
            # stride = 1 : kernel_size = 2 * padding + 1  -> 保持不变
            nn.Conv2d(inInputDim * inMultiple * inMultiple, inInputDim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, inData):
        return self.Blocks(inData)

class PixelShuffle_2(PixelShuffle):
    def __init__(self, inInputDim) -> None:
        super().__init__(inInputDim, 2)

########################################

class Conv_2(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()
        
        # stride = 2 : kernel_size = 2 * padding + 2 -> 缩小至二分之一
        self.Blocks = nn.Conv2d(inInputDim, inInputDim, kernel_size=2, stride=2, padding=0)

    def forward(self, inData):
        return self.Blocks(inData)

########################################

class Fused(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out
