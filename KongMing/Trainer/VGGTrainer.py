import torch

from .SingleNNTrainer import SingleNNTrainer

from KongMing.Models.BaseNNModel import BaseNNModel

import torch.nn.functional as F

class VGGTrainer(SingleNNTrainer) :
    def __init__(
            self,
            inNNModel : BaseNNModel,
            inLearningRate,
            inLogRootPath
        ) -> None:
        super().__init__(
            inLearningRate,
            inLogRootPath
        )
        self.NNModel        = inNNModel.to(self.Device)

    def _CreateOptimizer(self) -> None:
        self.NNModel.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))

    def _CreateLossFN(self) -> None:
        self.NNModel.ApplyLossFunc(F.smooth_l1_loss)


    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :
        # get BatchSize
        nBatchSize = inBatchData.size(0)
        
        # Prepare Real and Fake Data
        RealData = inBatchData.to(self.Device)
        
        with self.NNModel as Model:
            pass
