import torch
from torch import nn
import torch.nn.functional as F

from math import sqrt

class Up_2(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()

        self.Blocks = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, inData):
        return self.Blocks(inData)

class UpConv_2(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()

        self.Blocks = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            # kernel_size=3, stride=1, padding=1 保持输入大小变
            nn.Conv2d(inInputDim, inInputDim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, inData):
        return self.Blocks(inData)

class ConvT_2(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()

        self.Blocks = nn.ConvTranspose2d(inInputDim, inInputDim, kernel_size=4, stride=2, padding=1)

    def forward(self, inData):
        return self.Blocks(inData)


class Fused(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
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

        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out

