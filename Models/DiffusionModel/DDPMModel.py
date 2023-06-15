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

from .DiffusionModelUtils import Unet

class DMModel(Unet):
    def __init__(self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True,
            resnet_block_groups=8,
            use_convnext=True,
            convnext_mult=2,
            inTimesteps : int = 1000,
        ) -> None:
        super().__init__(dim, init_dim, out_dim, dim_mults, channels, with_time_emb, resnet_block_groups, use_convnext, convnext_mult)

        self.Timesteps             = inTimesteps
        
        Betas                      = BetaSchedule.Linear(self.Timesteps)
        
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
        Out = inData.gather(-1, inIndex)

        return Out.reshape(nBatchSize, *((1,) * (len(inShape) - 1))).to(inIndex.device)
    
    #这个函数用于生成噪声样本
    def Q_Sample(self, inXStart, inT, inNoise):
        XStartShape                 = inXStart.shape
        SqrtAlphasCumprod_T         = self.Extract(self.SqrtAlphasCumprod, inT, XStartShape)
        SqrtOneMinusAlphasCumprod_T = self.Extract(self.SqrtOneMinusAlphasCumprod, inT, XStartShape)
        #print("QSampleDevice:XStart:{}, Noise:{}, SQACT:{}, SQOMACT:{}".format(inXStart.device, inNoise.device, SqrtAlphasCumprod_T.device, SqrtOneMinusAlphasCumprod_T.device))
        return SqrtAlphasCumprod_T * inXStart + SqrtOneMinusAlphasCumprod_T * inNoise

    #这个函数用于生成噪声样本
    def P_Losses(self, inDenoiseModel, inXstart, inT, inNoise=None, inLossType="l1"):
        if inNoise is None:
            inNoise = torch.randn_like(inXstart)

        XNoisy = self.Q_Sample(inXStart=inXstart, inT=inT, inNoise=inNoise)
        PredictedNoise = inDenoiseModel(XNoisy, inT)

        if inLossType == 'l1':
            LossResult = F.l1_loss(inNoise, PredictedNoise)
        elif inLossType == 'l2':
            LossResult = F.mse_loss(inNoise, PredictedNoise)
        elif inLossType == "huber":
            LossResult = F.smooth_l1_loss(inNoise, PredictedNoise)
        else:
            raise NotImplementedError()

        return LossResult

    #这些函数用于在给定的时间点和下一时间点之间进行采样。
    def P_Sample(self, inModel, x, t, t_index):
        BetasT = self.Extract(self.Betas, t, x.shape)
        SqrtOneMinusAlphasCumprodT = self.Extract(
            self.SqrtOneMinusAlphasCumprod, t, x.shape
        )
        SqrtRecipAlphasT = self.Extract(self.SqrtRecipAlphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        ModelMean = SqrtRecipAlphasT * (
            x - BetasT * inModel(x, t) / SqrtOneMinusAlphasCumprodT
        )

        if t_index == 0:
            return ModelMean
        else:
            posterior_variance_t = self.Extract(self.PosteriorVariance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return ModelMean + torch.sqrt(posterior_variance_t) * noise 
    
    @torch.no_grad()
    def P_Sample_Loop(self, model, inShape):
        device = next(model.parameters()).device

        b = inShape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(inShape, device=device)
        imgs = []

        for i in reversed(range(0, self.Timesteps)):
            img = self.P_Sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu().numpy())
        return imgs

    # 函数入口
    @torch.no_grad()
    def Sample(self, inModel, inImageSize, inBatchSize=16, inChannels=3):
        return self.P_Sample_Loop(inModel, inShape=(inBatchSize, inChannels, inImageSize, inImageSize))

class DDPMModel(BaseModel) :
    def __init__(self, inImageSize, inChannel, inLearningRate=0.00001, inTimesteps : int = 1000, inModeRootlFolderPath="."):

        self.DMModel    = DMModel(dim=inImageSize, channels=inChannel, dim_mults=(1,2,4,),  inTimesteps=inTimesteps)

        NewTrainer      = DDPMTrainer(self.DMModel, inLearningRate, inTimesteps=inTimesteps)
        NewArchiver     = DDPMArchiver(self.DMModel, "DDPM", inModeRootlFolderPath)
        super().__init__(NewTrainer, NewArchiver)

    def Eval(self, *inArgs, **inKWArgs):
        if (super().Eval(*inArgs, **inKWArgs) == False) :
            return None
        self.DMModel.eval()
        return self.DMModel.Sample(
            self.DMModel,
            inImageSize=inKWArgs["inImageSize"],
            inBatchSize=inKWArgs["inBatchSize"],
            inChannels=inKWArgs["inChannels"]
        )
