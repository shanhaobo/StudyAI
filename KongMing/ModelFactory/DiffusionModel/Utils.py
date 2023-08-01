import torch

"""
====================================================================================
"""
class BetaSchedule() :
    def __init__(self) -> None:
        pass

    @staticmethod
    def Linear(timesteps:int, beta_start = 0.0001, beta_end = 0.02):
        return torch.linspace(beta_start, beta_end, steps=timesteps)

    @staticmethod
    def Quadratic(timesteps:int, beta_start = 0.0001, beta_end = 0.02):
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

    @staticmethod
    def Sigmoid(timesteps:int, beta_start = 0.0001, beta_end = 0.02):
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    @staticmethod
    def Cosine(inTimesteps:int, inS=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        Steps = inTimesteps + 1
        Schedule = torch.linspace(0, inTimesteps, Steps)
        AlphasCumprod = torch.cos(((Schedule / inTimesteps) + inS) / (1 + inS) * torch.pi * 0.5) ** 2
        AlphasCumprod = AlphasCumprod / AlphasCumprod[0]
        betas = 1 - (AlphasCumprod[1:] / AlphasCumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
 

 