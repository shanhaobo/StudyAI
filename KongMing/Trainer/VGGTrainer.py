import torch

from .SingleNNTrainer import SingleNNTrainer

from KongMing.Models.BaseNNModel import BaseNNModel

import torch.nn.functional as F
from torch import nn

class VGGTrainer(SingleNNTrainer) :
    def __init__(
            self,
            inNNModel : BaseNNModel,
            inLearningRate,
            inLogRootPath
        ) -> None:
        super().__init__(
            inNNModel,
            inLearningRate,
            inLogRootPath
        )

        self.EndEpochTrain.add(self.__VGGEndEpochTrain)

    def _CreateOptimizer(self) -> None:
        self.NNModel.ApplyOptimizer(torch.optim.SGD, self.LearningRate, momentum=0.9)
        self.NNModel.ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)

    def _CreateLossFN(self) -> None:
        self.NNModel.ApplyLossFunc(nn.CrossEntropyLoss().to(self.Device))

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :
        # Prepare Real and Fake Data
        DeviceData = inBatchData.to(self.Device)
        DeviceLabel = inBatchLabel.to(self.Device)
        
        with self.NNModel as Model:
            Output = Model(DeviceData)
            Model.CalcAndAcceptLoss(Output, DeviceLabel)

    def __VGGEndEpochTrain(self, inArgs, inKVArgs):
        self.NNModel.UpdateLRScheduler()
