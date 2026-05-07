import torch
from torch import nn

from .SingleNNTrainer import SingleNNTrainer


class MathNetTrainer(SingleNNTrainer):
    def __init__(self, inLearningRate : float) -> None:
        super().__init__(inLearningRate)

    def _CreateOptimizer(self) -> None:
        self.NNModel.ApplyOptimizer(torch.optim.Adam, self.LearningRate)
        # 和 VGGTrainer 一样小幅指数衰减；合成数据上其实可有可无，留个示范
        self.NNModel.ApplyLRScheduler(torch.optim.lr_scheduler.ExponentialLR, gamma=0.999)

    def _CreateLossFN(self) -> None:
        self.NNModel.ApplyLossFunc(nn.MSELoss().to(self.Device))

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) -> None:
        DeviceData  = inBatchData.to(self.Device)
        DeviceLabel = inBatchLabel.to(self.Device)

        with self.NNModel as Model:
            Output = Model(DeviceData)
            Model.CalcAndAcceptLoss(Output, DeviceLabel)
