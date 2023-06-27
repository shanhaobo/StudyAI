
import torch

import torch.nn.functional as F
from einops import reduce
from functools import partial

class WeightStandardizedConv2D(torch.nn.Conv2d):
    def forward(self, x):
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + 1e-5).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
