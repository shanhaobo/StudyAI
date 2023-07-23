import torch

from datetime import datetime

from .BaseTrainer import BaseTrainer

from KongMing.Models.BaseNNModel import BaseNNModel

class SingleNNTrainer(BaseTrainer) :
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

        self.EndBatchTrain.add(self.__SNNEndBatchTrain)
        self.EndEpochTrain.add(self.__SNNEndEpochTrain)

    def __SNNEndBatchTrain(self, inArgs, inKVArgs) -> None:
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

    def __SNNEndEpochTrain(self, inArgs, inKVArgs):
        self.NNModel.UpdateLRScheduler()
