
import torch

import torch.nn.functional as F
from einops import reduce
from functools import partial

class WeightStandardizedConv2D(torch.nn.Conv2d):
    def forward(self, inX):
        Weight = self.weight
        Mean = reduce(Weight, "o ... -> o 1 1 1", "mean")
        Var = reduce(Weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        NormalizedWeight = (Weight - Mean) * (Var + 1e-5).rsqrt()

        return F.conv2d(
            inX,
            NormalizedWeight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
