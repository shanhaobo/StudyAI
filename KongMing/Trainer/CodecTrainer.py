import torch

from typing import Dict as TypedDict

from KongMing.Models.BaseNNModel import BaseNNModel

from .MultiNNTrainer import MultiNNTrainer

class CodecTrainer(MultiNNTrainer) :
    def __init__(self,
            inLearningRate,
            inLogRootPath
        ) -> None:
        super().__init__(inLearningRate, inLogRootPath)

    def RegisterMultiNNModule(
            self,
            inNNModelDict : TypedDict[str, torch.nn.Module]
        ) -> None:
        super().RegisterMultiNNModule(inNNModelDict)

        self.Encoder : BaseNNModel  = self.NNModuleDict["Encoder"]
        self.Decoder : BaseNNModel  = self.NNModuleDict["Decoder"]

    def _CreateOptimizer(self) -> None:
        pass
    
    def _CreateLossFN(self) -> None:
        pass

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :
        # get BatchSize
        nBatchSize = inBatchData.size(0)
        
        # Prepare Real and Fake Data
        RealData = inBatchData.to(self.Device)
        
        with self.Decoder as D:
            with self.Encoder as E:
                pass
            pass
