import torch
import os
from datetime import datetime

import torch.nn.functional as F

from .BaseTrainer import BaseTrainer

from KongMing.Modules.BaseNNModule import BaseNNModule

import pandas as pd

class DDPMTrainer(BaseTrainer) :
    def __init__(self, 
            inNN : BaseNNModule,
            inDiffusionMode : BaseNNModule,
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

        self.LossData       = {"Epoch":[], "Batch":[], "Loss":[], "AvgLoss":[]}

###########################################################################################

    def _CreateOptimizer(self) -> None:
        #self.Optimizer = torch.optim.Adam(self.NNModel.parameters(),     lr=self.LearningRate, betas=(0.5, 0.999))
        self.NNModel.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))

    def _CreateLossFN(self) -> None:
        self.NNModel.ApplyLossFunc(F.smooth_l1_loss)

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :
        # get BatchSize
        nBatchSize          = inBatchData.size(0)
        RealData            = inBatchData.to(self.Device)

        with self.NNModel as Model:
            Noise               = torch.randn_like(RealData)

            TimeEmbedding       = torch.randint(0, self.Timesteps, (nBatchSize,), device=self.Device).long()
            RealDataWithNoise   = self.DiffusionMode.Q_Sample(inXStart = RealData, inT = TimeEmbedding, inNoise = Noise)

            PredictedNoise      = Model(RealDataWithNoise, TimeEmbedding)
            
            Model.CalcAndAcceptLoss(Noise, PredictedNoise)

###########################################################################################

    def DDPMEndBatchTrain(self, inArgs, inKVArgs) -> None:
        Loss, AvgLoss = self.NNModel.GetLoss()
        
        print(
            "{} | Epoch:{:0>4d} | Batch:{:0>4d} | Loss:{:.8f} | AverageLoss:{:.8f}".
            format(
                datetime.now().strftime("[%Y/%m/%d %H:%M:%S.%f]"),
                self.CurrEpochIndex,
                self.CurrBatchIndex,
                Loss,
                AvgLoss
            )
        )
        self.LossData["Epoch"].append(self.CurrEpochIndex)
        self.LossData["Batch"].append(self.CurrBatchIndex)
        self.LossData["Loss"].append(Loss)
        self.LossData["AvgLoss"].append(AvgLoss)


    def DDPMEndEpochTrain(self, inArgs, inKVArgs) -> None:
        df = pd.DataFrame(self.LossData)
        os.makedirs(self.LogRootPath, exist_ok=True)
        df.to_csv("{}/loss.csv".format(self.LogRootPath), mode='a', index=False, header=False)
        self.LossData["Epoch"].clear()
        self.LossData["Batch"].clear()
        self.LossData["Loss"].clear()
        self.LossData["AvgLoss"].clear()

    def DDPMBeginTrain(self, inArgs, inKVArgs) -> None:
        if "ema_override" in inArgs:
            print("EMA Override........")
            self.DiffusionMode.EMA.override_parameters(self.NNModel)

    def _CheckEndEpoch(self)->bool:
        _, AvgLoss = self.NNModel.GetLoss()
        return AvgLoss <= 0.01
###########################################################################################
