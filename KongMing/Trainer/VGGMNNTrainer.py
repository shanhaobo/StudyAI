import torch

from datetime import datetime

from .MultiNNTrainer import MultiNNTrainer

from KongMing.Models.BaseNNModel import BaseNNModel

import torch.nn.functional as F
from torch import nn

from typing import Dict as TypedDict

class VGGMNNTrainer(MultiNNTrainer) :
    def __init__(
            self,
            inLearningRate,
            inLogRootPath
        ) -> None:
        super().__init__(
            inLearningRate,
            inLogRootPath
        )

        self.EndBatchTrain.add(self.__VGGMNNEndBatchTrain)
        
    def RegisterMultiNNModule(
            self,
            inNNModelDict : TypedDict[str, torch.nn.Module]
        ) -> None:
        super().RegisterMultiNNModule(inNNModelDict)

        self.VGG1 : BaseNNModel  = self.NNModuleDict["VGG1"]
        self.VGG2 : BaseNNModel  = self.NNModuleDict["VGG2"]
        self.VGG3 : BaseNNModel  = self.NNModuleDict["VGG3"]
        self.VGG4 : BaseNNModel  = self.NNModuleDict["VGG4"]
        self.VGG5 : BaseNNModel  = self.NNModuleDict["VGG5"]

    def _CreateOptimizer(self) -> None:
        self.VGG1.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.VGG1.ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        self.VGG2.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.VGG2.ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        self.VGG3.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.VGG3.ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        self.VGG4.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.VGG4.ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        self.VGG5.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.VGG5.ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        #self.NNModel.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        #self.NNModel.ApplyOptimizer(torch.optim.SGD, self.LearningRate, momentum=0.9)
        #self.NNModel.ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        pass

    def _CreateLossFN(self) -> None:
        self.VGG1.ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        self.VGG2.ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        self.VGG3.ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        self.VGG4.ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        self.VGG5.ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        #self.NNModel.ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        pass

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :
        # Prepare Real and Fake Data
        DeviceData = inBatchData.to(self.Device)
        DeviceLabel = inBatchLabel.to(self.Device)
        
        with self.VGG1 as Model:
            MidData, Output = Model(DeviceData)
            Model.CalcAndAcceptLoss(Output, DeviceLabel)

        with self.VGG2 as Model:
            MidData, Output = Model(MidData.detach())
            Model.CalcAndAcceptLoss(Output, DeviceLabel)
        
        with self.VGG3 as Model:
            MidData, Output = Model(MidData.detach())
            Model.CalcAndAcceptLoss(Output, DeviceLabel)
            
        with self.VGG4 as Model:
            MidData, Output = Model(MidData.detach())
            Model.CalcAndAcceptLoss(Output, DeviceLabel)

        with self.VGG5 as Model:
            Output = Model(MidData.detach())
            Model.CalcAndAcceptLoss(Output, DeviceLabel)

    def __VGGMNNEndBatchTrain(self, inArgs, inKVArgs) -> None:
        Loss, AvgLoss = self.VGG5.GetLossValue()

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
