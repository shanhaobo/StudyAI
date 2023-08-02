import torch
import os
from datetime import datetime

import torch.nn.functional as F

from .MultiNNTrainer import MultiNNTrainer

from typing import Dict as TypedDict

from KongMing.Models.BaseNNModel import BaseNNModel

import pandas as pd

class DDPMTrainer(MultiNNTrainer) :
    def __init__(self,
            inLearningRate,
            inLogRootPath
        ) -> None:
        super().__init__(inLearningRate)
        self.LogRootPath = inLogRootPath

        self.BeginTrain.add(self.DDPMBeginTrain)

        self.EndBatchTrain.add(self.DDPMEndBatchTrain)
        self.EndEpochTrain.add(self.DDPMEndEpochTrain)

        self.LossData       = {"Epoch":[], "Batch":[], "Loss":[], "AvgLoss":[]}

    def RegisterMultiNNModule(
            self,
            inNNModelDict : TypedDict[str, torch.nn.Module]
        ) -> None:
        super().RegisterMultiNNModule(inNNModelDict)

        self.NNModel : BaseNNModel  = self.NNModuleDict["NNModel"]
        self.DiffusionMode          = self.NNModuleDict["DiffusionModel"]

###########################################################################################

    def _CreateOptimizer(self) -> None:
        self.NNModel.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))

    def _CreateLossFN(self) -> None:
        self.NNModel.ApplyLossFunc(F.smooth_l1_loss)

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :
        # get BatchSize
        nBatchSize          = inBatchData.size(0)
        RealData            = inBatchData.to(self.Device)

        with self.NNModel as Model:
            Noise               = torch.randn_like(RealData)

            TimeEmbedding       = torch.randint(0, self.DiffusionMode.Timesteps, (nBatchSize,), device=self.Device).long()
            RealDataWithNoise   = self.DiffusionMode.Q_Sample(inXStart = RealData, inT = TimeEmbedding, inNoise = Noise)

            PredictedNoise      = Model(RealDataWithNoise, TimeEmbedding)
            
            Model.CalcAndAcceptLoss(Noise, PredictedNoise)

###########################################################################################

    def DDPMEndBatchTrain(self, inArgs, inKVArgs) -> None:
        Loss, AvgLoss = self.NNModel.GetLossValue()
        
        print(
            "{} | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.6f} / {:.6f}".
            format(
                datetime.now().strftime("[%Y/%m/%d %H:%M:%S.%f]"),
                self.CurrEpochIndex,
                self.EndEpochIndex,
                self.CurrBatchIndex + 1,
                self.BatchNumPerEpoch,
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
            self.DiffusionMode.EMA.OverrideParameters(self.NNModel)

    def _CheckEndEpoch(self)->bool:
        _, AvgLoss = self.NNModel.GetLossValue()
        return AvgLoss <= 0.01
###########################################################################################
