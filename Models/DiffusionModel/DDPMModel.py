import torch
from torch.utils.data import DataLoader
from torch import nn, einsum
import torch.nn.functional as F

from datetime import datetime

import math
from inspect import isfunction
from functools import partial

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from Archiver.BaseArchiver import BaseArchiver
from Models.BaseModel import BaseModel
from Trainer.BaseTrainer import BaseTrainer

import torch.nn.functional as F

from .Utils import BetaSchedule

class DDPMModel(BaseModel) :
    def __init__(self, inTrainer: BaseTrainer, inArchiver: BaseArchiver, inTimesteps : int = 1000):
        super().__init__(inTrainer, inArchiver)

        self.Betas                      = BetaSchedule.Linear(inTimesteps)
        
        self.Alphas                     = 1 - self.Betas
        self.AlphasCumprod              = torch.cumprod(self.Alphas, axis = 0)
        self.AlphasCumprodPrev          = F.pad(self.AlphasCumprod[:-1], (1, 0), value=1.0)
        self.SqrtRecipAlphas            = torch.sqrt(1.0 / self.Alphas)

        self.SqrtAlphasCumprod          = torch.sqrt(self.AlphasCumprod)
        self.SqrtOneMinusAlphasCumprod  = torch.sqrt(1 - self.AlphasCumprod)

        self.PosteriorVariance          = self.Betas * (1.0 - self.AlphasCumprodPrev) / (1.0 - self.AlphasCumprod)

        print(self.Betas)
        print(self.Alphas)
        print(self.SqrtRecipAlphas)
        print(self.AlphasCumprod)
        print(self.SqrtAlphasCumprod)
        print(self.SqrtOneMinusAlphasCumprod)
        print(self.AlphasCumprodPrev)
        print(self.PosteriorVariance)
