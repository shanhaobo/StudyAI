import torch

from datetime import datetime

import torch.nn.functional as F

from .BaseTrainer import BaseTrainer

from KongMing.Models.Zoo.EMA import EMA

class DDPMTrainer(BaseTrainer) :
    def __init__(self, 
            inNN : torch.nn.Module,
            inDiffusionMode : torch.nn.Module,
            inLearningRate,
            inTimesteps = 1000
        ) -> None:
        super().__init__(inLearningRate)
        self.NNModel        = inNN.to(self.Device)
        self.DiffusionMode  = inDiffusionMode.to(self.Device)

        self.Timesteps      = inTimesteps

        self.EMA            = EMA(inNN, 0.999)

        self.EndBatchTrain.add(self.DDPMEndBatchTrain)

        self.SumLoss        = 0
        self.LastBatch      = 1

###########################################################################################

    def _CreateOptimizer(self) -> None:
        self.Optimizer = torch.optim.Adam(self.NNModel.parameters(),     lr=self.LearningRate, betas=(0.5, 0.999))
        pass

    def _CreateLossFN(self) -> None:
        LossType = "huber"
        if LossType == "L1":
            self.LossFN = F.l1_loss
        elif LossType == "L2":
            self.LossFN = F.mse_loss
        else:
            self.LossFN = F.smooth_l1_loss
        pass

    def _BatchTrain(self, inBatchData, inBatchLabel, *inArgs, **inKWArgs) :
        # get BatchSize
        nBatchSize          = inBatchData.size(0)
        RealData            = inBatchData.to(self.Device)
        Noise               = torch.randn_like(RealData)

        TimeEmbedding       = torch.randint(0, self.Timesteps, (nBatchSize,), device=self.Device).long()
        RealDataWithNoise   = self.DiffusionMode.Q_Sample(inXStart = RealData, inT = TimeEmbedding, inNoise = Noise)

        PredictedNoise      = self.NNModel(RealDataWithNoise, TimeEmbedding)

        #loss = self.LossFN(PredictedNoise, RealData)
        loss = self.LossFN(Noise, PredictedNoise)

        self._BackPropagate(self.Optimizer, loss)

        self.CurrBatchDDPMLoss = loss.item()

        self.SumLoss        += self.CurrBatchDDPMLoss
        self.LastBatch      = self.CurrBatchIndex + 1

###########################################################################################

    def DDPMEndBatchTrain(self, *inArgs, **inKWArgs) -> None:
        NowStr  = datetime.now().strftime("[%Y/%m/%d %H:%M:%S.%f]")
        print(
            "{} | Epoch:{:0>4d} | Batch:{:0>6d} | Loss:{:.8f}".
            format(
                NowStr,
                self.CurrEpochIndex,
                self.CurrBatchIndex,
                self.CurrBatchDDPMLoss,
            )
        )

    def __Continue(self)->bool:
        AverageLoss = self.SumLoss / self.LastBatch

        return AverageLoss > 0.01
###########################################################################################
