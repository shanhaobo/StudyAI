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

from ..Moduels.UNet2D import UNet2D

class DMModel(nn.Module):
    def __init__(self, inTimesteps : int = 1000) -> None:
        super().__init__()
        
        Betas                      = BetaSchedule.Linear(inTimesteps)
        
        Alphas                     = 1 - Betas
        AlphasCumprod              = torch.cumprod(Alphas, axis = 0)
        AlphasCumprodPrev          = F.pad(AlphasCumprod[:-1], (1, 0), value=1.0)
        SqrtRecipAlphas            = torch.sqrt(1.0 / Alphas)

        SqrtAlphasCumprod          = torch.sqrt(AlphasCumprod)
        SqrtOneMinusAlphasCumprod  = torch.sqrt(1 - AlphasCumprod)

        PosteriorVariance          = Betas * (1.0 - AlphasCumprodPrev) / (1.0 - AlphasCumprod)

        RegisterBufferF32 = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        RegisterBufferF32("Betas", Betas)

        RegisterBufferF32("Alphas", Alphas)
        RegisterBufferF32("AlphasCumprod", AlphasCumprod)
        RegisterBufferF32("AlphasCumprodPrev", AlphasCumprodPrev)

        RegisterBufferF32("SqrtRecipAlphas", SqrtRecipAlphas)
        RegisterBufferF32("SqrtAlphasCumprod", SqrtAlphasCumprod)
        RegisterBufferF32("SqrtOneMinusAlphasCumprod", SqrtOneMinusAlphasCumprod)

        RegisterBufferF32("PosteriorVariance", PosteriorVariance)

    @staticmethod
    def Extract(inData, inIndex, inShape):
        nBatchSize = inIndex.shape[0]
        Out = inData.gather(-1, inIndex.cpu())

        return Out.reshape(nBatchSize, *((1,) * (len(inShape) - 1))).to(inIndex.device)

class DDPMModel(BaseModel) :
    def __init__(self, inTrainer: BaseTrainer, inArchiver: BaseArchiver, inTimesteps : int = 1000):
        super().__init__(inTrainer, inArchiver)

        self.ModelFrame = UNet2D(3, 10)
        self.DMModel    = DMModel(inTimesteps)
