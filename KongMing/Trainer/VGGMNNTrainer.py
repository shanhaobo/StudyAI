import torch

from datetime import datetime

from .MultiNNTrainer import MultiNNTrainer

from KongMing.Models.BaseNNModel import BaseNNModel

import torch.nn.functional as F
from torch import nn

class VGGMNNTrainer(MultiNNTrainer) :
    def __init__(
            self,
            inMNNDict,
            inLearningRate,
            inLogRootPath
        ) -> None:
        super().__init__(
            inMNNDict,
            inLearningRate,
            inLogRootPath
        )

        self.EndBatchTrain.add(self.__VGGMNNEndBatchTrain)
        

    def _CreateOptimizer(self) -> None:
        self.NNModuleDict["VGG1"].ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.NNModuleDict["VGG1"].ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        self.NNModuleDict["VGG2"].ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.NNModuleDict["VGG2"].ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        self.NNModuleDict["VGG3"].ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.NNModuleDict["VGG3"].ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        self.NNModuleDict["VGG4"].ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.NNModuleDict["VGG4"].ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        self.NNModuleDict["VGG5"].ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.NNModuleDict["VGG5"].ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        #self.NNModel.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        #self.NNModel.ApplyOptimizer(torch.optim.SGD, self.LearningRate, momentum=0.9)
        #self.NNModel.ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        pass

    def _CreateLossFN(self) -> None:
        self.NNModuleDict["VGG1"].ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        self.NNModuleDict["VGG2"].ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        self.NNModuleDict["VGG3"].ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        self.NNModuleDict["VGG4"].ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        self.NNModuleDict["VGG5"].ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        #self.NNModel.ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        pass

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :
        # Prepare Real and Fake Data
        DeviceData = inBatchData.to(self.Device)
        DeviceLabel = inBatchLabel.to(self.Device)
        
        with self.NNModuleDict["VGG1"] as Model:
            MidData, Output = Model(DeviceData)
            Model.CalcAndAcceptLoss(Output, DeviceLabel)

        with self.NNModuleDict["VGG2"] as Model:
            MidData, Output = Model(MidData.detach())
            Model.CalcAndAcceptLoss(Output, DeviceLabel)
        
        with self.NNModuleDict["VGG3"] as Model:
            MidData, Output = Model(MidData.detach())
            Model.CalcAndAcceptLoss(Output, DeviceLabel)
            
        with self.NNModuleDict["VGG4"] as Model:
            MidData, Output = Model(MidData.detach())
            Model.CalcAndAcceptLoss(Output, DeviceLabel)

        with self.NNModuleDict["VGG5"] as Model:
            Output = Model(MidData.detach())
            Model.CalcAndAcceptLoss(Output, DeviceLabel)

        """
        MidData, Output = self.MNNDict["VGG1"](DeviceData)
        self.MNNDict["VGG1"].CalcAndAcceptLoss(Output, DeviceLabel)
        self.MNNDict["VGG1"].BackPropagate()

        MidData, Output = self.MNNDict["VGG2"](MidData)
        self.MNNDict["VGG2"].CalcAndAcceptLoss(Output, DeviceLabel)
        self.MNNDict["VGG2"].BackPropagate()
        """

    def __VGGMNNEndBatchTrain(self, inArgs, inKVArgs) -> None:
        Loss, AvgLoss = self.NNModuleDict["VGG5"].GetLossValue()

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
