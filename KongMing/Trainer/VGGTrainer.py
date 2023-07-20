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

    def _CreateOptimizer(self) -> None:
        self.NNModel.ApplyOptimizer(torch.optim.SGD, lr=self.LearningRate, momentum=0.9)

    def _CreateLossFN(self) -> None:
        self.NNModel.ApplyLossFunc(nn.CrossEntropyLoss)

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :
        # get BatchSize
        nBatchSize = inBatchData.size(0)
        
        # Prepare Real and Fake Data
        DeviceData = inBatchData.to(self.Device)
        DeviceLabel = inBatchLabel.to(self.Device)
        
        with self.NNModel as Model:
            Output = Model(DeviceData)
            Model.CalcAndAcceptLoss(Output, DeviceLabel)
