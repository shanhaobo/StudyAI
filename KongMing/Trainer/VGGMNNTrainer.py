import torch

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
            inLearningRate,
            inLogRootPath
        )
        self.MNNDict = inMNNDict

    def _CreateOptimizer(self) -> None:
        self.MNNDict["VGG1"].ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.MNNDict["VGG1"].ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        self.MNNDict["VGG2"].ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.MNNDict["VGG2"].ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        self.MNNDict["VGG3"].ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.MNNDict["VGG3"].ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        self.MNNDict["VGG4"].ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.MNNDict["VGG4"].ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        self.MNNDict["VGG5"].ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.MNNDict["VGG5"].ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        #self.NNModel.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        #self.NNModel.ApplyOptimizer(torch.optim.SGD, self.LearningRate, momentum=0.9)
        #self.NNModel.ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)
        pass

    def _CreateLossFN(self) -> None:
        self.MNNDict["VGG1"].ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        self.MNNDict["VGG2"].ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        self.MNNDict["VGG3"].ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        self.MNNDict["VGG4"].ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        self.MNNDict["VGG5"].ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        #self.NNModel.ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))
        pass

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :
        # Prepare Real and Fake Data
        DeviceData = inBatchData.to(self.Device)
        DeviceLabel = inBatchLabel.to(self.Device)
        
        with self.MNNDict["VGG1"] as Model:
            Output = Model(DeviceData)
            Model.CalcAndAcceptLoss(Output, DeviceLabel)

        with self.MNNDict["VGG2"] as Model:
            Output = Model(DeviceData)
            Model.CalcAndAcceptLoss(Output, DeviceLabel)
        
        with self.MNNDict["VGG3"] as Model:
            Output = Model(DeviceData)
            Model.CalcAndAcceptLoss(Output, DeviceLabel)
            
        with self.MNNDict["VGG4"] as Model:
            Output = Model(DeviceData)
            Model.CalcAndAcceptLoss(Output, DeviceLabel)

        with self.MNNDict["VGG5"] as Model:
            Output = Model(DeviceData)
            Model.CalcAndAcceptLoss(Output, DeviceLabel)