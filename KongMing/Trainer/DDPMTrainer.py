import torch
import os
from datetime import datetime

import torch.nn.functional as F

from .BaseTrainer import BaseTrainer

import pandas as pd

class DDPMTrainer(BaseTrainer) :
    def __init__(self, 
            inNN : torch.nn.Module,
            inDiffusionMode : torch.nn.Module,
            inLearningRate,
            inTimesteps = 1000,
            inLogRootPath = "."
        ) -> None:
        super().__init__(inLearningRate, inLogRootPath)
        self.NNModel        = inNN.to(self.Device)
        self.DiffusionMode  = inDiffusionMode.to(self.Device)

        self.Timesteps      = inTimesteps

        self.BeginTrain.add(self.DDPMBeginTrain)

        self.EndBatchTrain.add(self.DDPMEndBatchTrain)
        self.EndEpochTrain.add(self.DDPMEndEpochTrain)

        self.SumLoss        = 1
        self.LastBatch      = 1

        self.LossData       = {"Epoch":[], "Batch":[], "Loss":[], "AvgLoss":[]}

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
        self.DiffusionMode.EMA.update_parameters(self.NNModel)

        self.CurrBatchDDPMLoss = loss.item()

        self.SumLoss        += self.CurrBatchDDPMLoss
        self.LastBatch      = self.CurrBatchIndex + 1

###########################################################################################

    def DDPMEndBatchTrain(self, *inArgs, **inKWArgs) -> None:
        NowStr  = datetime.now().strftime("[%Y/%m/%d %H:%M:%S.%f]")
        print(
            "{} | Epoch:{:0>4d} | Batch:{:0>6d} | Loss:{:.8f} | AverageLoss:{:.8f}".
            format(
                NowStr,
                self.CurrEpochIndex,
                self.CurrBatchIndex,
                self.CurrBatchDDPMLoss,
                self.SumLoss / (self.CurrBatchIndex + 1)
            )
        )
        self.LossData["Epoch"].append(self.CurrEpochIndex)
        self.LossData["Batch"].append(self.CurrBatchIndex)
        self.LossData["Loss"].append(self.CurrBatchDDPMLoss)
        self.LossData["AvgLoss"].append(self.SumLoss / (self.CurrBatchIndex + 1))


    def DDPMEndEpochTrain(self, *inArgs, **inKWArgs) -> None:
        df = pd.DataFrame(self.LossData)
        os.makedirs(self.LogRootPath, exist_ok=True)
        df.to_csv("{}/loss.csv".format(self.LogRootPath), mode='a', index=False)
        self.LossData["Epoch"].clear()
        self.LossData["Batch"].clear()
        self.LossData["Loss"].clear()
        self.LossData["AvgLoss"].clear()

    def DDPMBeginTrain(self, *inArgs, **inKWArgs) -> None:
        OverrideEMA = inKWArgs.get("ema_override")
        if OverrideEMA is not None and OverrideEMA == "true":
            self.DiffusionMode.EMA.override_parameters(self.NNModel)

    def _Continue(self)->bool:
        AverageLoss = self.SumLoss / self.LastBatch
        Result =  AverageLoss > 0.01
        self.SumLoss = 0
        return Result
###########################################################################################
