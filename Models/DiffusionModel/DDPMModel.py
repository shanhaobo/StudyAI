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

from Models.BaseModel import BaseModel

from Archiver.DDPMArchiver import DDPMArchiver
from Trainer.DDPMTrainer import DDPMTrainer

import torch.nn.functional as F

from .Utils import BetaSchedule

from ..Moduels.UNet2D import UNet2D

from ..Zoo.EMA import EMA

class DMModel(nn.Module):
    def __init__(self, inTimesteps : int = 1000) -> None:
        super().__init__()

        self.Timesteps             = inTimesteps
        
        Betas                      = BetaSchedule.Linear(inTimesteps)
        
        Alphas                     = 1 - Betas
        AlphasCumprod              = torch.cumprod(Alphas, axis = 0)
        AlphasCumprodPrev          = F.pad(AlphasCumprod[:-1], (1, 0), value=1.0)
        SqrtRecipAlphas            = torch.sqrt(1.0 / Alphas)

        SqrtAlphasCumprod          = torch.sqrt(AlphasCumprod)
        SqrtOneMinusAlphasCumprod  = torch.sqrt(1 - AlphasCumprod)

        PosteriorVariance          = Betas * (1.0 - AlphasCumprodPrev) / (1.0 - AlphasCumprod)

        RegisterBufferF32 = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        RegisterBufferF32("Betas",                      Betas)

        RegisterBufferF32("Alphas",                     Alphas)
        RegisterBufferF32("AlphasCumprod",              AlphasCumprod)
        RegisterBufferF32("AlphasCumprodPrev",          AlphasCumprodPrev)

        RegisterBufferF32("SqrtRecipAlphas",            SqrtRecipAlphas)

        RegisterBufferF32("SqrtAlphasCumprod",          SqrtAlphasCumprod)
        RegisterBufferF32("SqrtOneMinusAlphasCumprod",  SqrtOneMinusAlphasCumprod)

        RegisterBufferF32("PosteriorVariance",          PosteriorVariance)

    @staticmethod
    def Extract(inData, inIndex, inShape):
        nBatchSize = inIndex.shape[0]
        Out = inData.gather(-1, inIndex.cpu())

        return Out.reshape(nBatchSize, *((1,) * (len(inShape) - 1))).to(inIndex.device)
    
    def Q_Sample(self, inXStart, inT, inNoise):
        XStartShape                 = inXStart.shape

        SqrtAlphasCumprod_T         = self.Extract(self.SqrtAlphasCumprod, inT, XStartShape)
        SqrtOneMinusAlphasCumprod_T = self.Extract(self.SqrtOneMinusAlphasCumprod, inT, XStartShape)
        return SqrtAlphasCumprod_T * inXStart + SqrtOneMinusAlphasCumprod_T * inNoise

    def P_Losses(self, denoise_model, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.Q_Sample(inXStart=x_start, inT=t, inNoise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def P_Sample(self, model, x, t, t_index):
        betas_t = self.Extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.Extract(
            self.SqrtOneMinusAlphasCumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.Extract(self.SqrtRecipAlphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.Extract(self.PosteriorVariance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
    
    @torch.no_grad()
    def P_Sample_Loop(self, model, shape):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in reversed(range(0, self.Timesteps)):
            img = self.P_Sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    # 函数入口
    @torch.no_grad()
    def Sample(self, model, image_size, batch_size=16, channels=3):
        return self.P_Sample_Loop(model, shape=(batch_size, channels, image_size, image_size))




class DDPMModel(BaseModel) :
    def __init__(self, inLearningRate=0.00001, inTimesteps : int = 1000, inModeRootlFolderPath="."):

        self.DMModel    = DMModel(inTimesteps)

        NewTrainer    = DDPMTrainer(self.DMModel, inLearningRate, inTimesteps=inTimesteps)
        NewArchiver   = DDPMArchiver("DDPM", inModeRootlFolderPath)
        super().__init__(NewTrainer, NewArchiver)

        self.ModelFrame = UNet2D(3, 10)
        self.DMModel    = DMModel(inTimesteps)

        self.EMA        = EMA(self.ModelFrame, 0.999)


