import torch
from torch import nn, einsum
import torch.nn.functional as F

import math
from inspect import isfunction
from functools import partial

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

"""
====================================================================================
"""
def Exists(x):
    return x is not None

def DefaultValue(val, d):
    if Exists(val):
        return val
    return d() if isfunction(d) else d


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, DefaultValue(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, DefaultValue(dim_out, dim), 1),
    )

"""
====================================================================================
"""
class BetaSchedule() :
    def __init__(self) -> None:
        pass

    @staticmethod
    def Linear(timesteps, beta_start = 0.0001, beta_end = 0.02):
        return torch.linspace(beta_start, beta_end, timesteps)

    @staticmethod
    def Quadratic(timesteps, beta_start = 0.0001, beta_end = 0.02):
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

    @staticmethod
    def Sigmoid(timesteps, beta_start = 0.0001, beta_end = 0.02):
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    @staticmethod
    def Cosine(timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

"""
====================================================================================
"""
def Extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
