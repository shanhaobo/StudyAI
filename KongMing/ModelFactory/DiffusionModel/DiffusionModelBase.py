import torch
import torch.nn.functional as F

from KongMing.Modules.AveragedModel import EMA

from KongMing.Modules.BaseNNModule import BaseNNModule

from .Utils import BetaSchedule

class DiffusionModel(BaseNNModule):
    def __init__(self, inTimesteps, inNNModule:torch.nn.Module) -> None:
        super().__init__()

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

        self.ApplyEMA(0.999, inNNModule)

    @staticmethod
    def Extract(inData, inIndex, inShape):
        nBatchSize = inIndex.shape[0]
        Out = inData.gather(-1, inIndex)

        return Out.reshape(nBatchSize, *((1,) * (len(inShape) - 1))).to(inIndex.device)
    
    #这个函数用于生成噪声样本
    def Q_Sample(self, inXStart, inT, inNoise):
        SqrtAlphasCumprod_T         = self.Extract(self.SqrtAlphasCumprod, inT, inXStart.shape)
        SqrtOneMinusAlphasCumprod_T = self.Extract(self.SqrtOneMinusAlphasCumprod, inT, inXStart.shape)
        #print("QSampleDevice:XStart:{}, Noise:{}, SQACT:{}, SQOMACT:{}".format(inXStart.device, inNoise.device, SqrtAlphasCumprod_T.device, SqrtOneMinusAlphasCumprod_T.device))
        return SqrtAlphasCumprod_T * inXStart + SqrtOneMinusAlphasCumprod_T * inNoise

    #这个函数用于生成噪声样本
    def P_Losses(self, inNNModel, inXstart, inT, inNoise=None, inLossType="l1"):
        if inNoise is None:
            inNoise = torch.randn_like(inXstart)

        XNoisy = self.Q_Sample(inXStart=inXstart, inT=inT, inNoise=inNoise)
        PredictedNoise = inNNModel(XNoisy, inT)

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
    def P_Sample(self, inNNModel, inXStart, inT, inIndexT):
        BetasT = self.Extract(self.Betas, inT, inXStart.shape)
        SqrtOneMinusAlphasCumprodT = self.Extract(
            self.SqrtOneMinusAlphasCumprod, inT, inXStart.shape
        )
        SqrtRecipAlphasT = self.Extract(self.SqrtRecipAlphas, inT, inXStart.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        ModelMean = SqrtRecipAlphasT * (
            inXStart - BetasT * inNNModel(inXStart, inT) / SqrtOneMinusAlphasCumprodT
        )

        if inIndexT == 0:
            return ModelMean
        else:
            PosteriorVarianceT = self.Extract(self.PosteriorVariance, inT, inXStart.shape)
            Noise = torch.randn_like(inXStart)
            # Algorithm 2 line 4:
            return ModelMean + torch.sqrt(PosteriorVarianceT) * Noise 
    
    @torch.no_grad()
    def P_Sample_Loop(self, inNNModel, inShape):
        Device = next(inNNModel.parameters()).device

        nBatchSize = inShape[0]
        # start from pure noise (for each example in the batch)
        Image = torch.randn(inShape, device=Device)

        for i in reversed(range(0, self.Timesteps)):
            Image = self.P_Sample(inNNModel, Image, torch.full((nBatchSize,), i, device=Device, dtype=torch.long), i)
            print("Sample Count : {}/{}".format(self.Timesteps - i, self.Timesteps))
        return Image

    # 函数入口
    @torch.no_grad()
    def Sample(self, inNNModel, inImageSize, inColorChanNum, inBatchSize):
        return self.P_Sample_Loop(inNNModel, inShape=(inBatchSize, inColorChanNum, inImageSize, inImageSize))

