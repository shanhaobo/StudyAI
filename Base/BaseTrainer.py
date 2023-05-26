import abc
import torch

from torch import Tensor
from torch.optim.optimizer import Optimizer

from torch.utils.data import DataLoader

class BaseTrainer(abc.ABC):
    def __init__(self, inLearningRate) -> None:
        self.Device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.LearningRate   = inLearningRate
        pass

    @staticmethod
    def _BackPropagate(inOptimizer : Optimizer, inLoss : Tensor) -> None:
        inOptimizer.zero_grad()
        inLoss.backward()
        inOptimizer.step()
    
    @abc.abstractmethod
    def _CreateOptimizer(self) -> None:
        pass

    @abc.abstractmethod
    def _CreateLossFN(self) -> None:
        pass

    @abc.abstractmethod
    def _BeginBatchTrain(self, inBatchIndex, **inArgs) -> None:
        pass

    @abc.abstractmethod
    def _BatchTrain(self, inBatchData) :
        pass
    
    @abc.abstractmethod
    def _EndBatchTrain(self, inBatchIndex, **inArgs) -> None:
        pass


    @abc.abstractmethod
    def _BeginEpochTrain(self, inEpochIndex, **inArgs) -> None:
        pass

    def _TrainEpoch(self, inEpochIndex, inDataLoader:DataLoader, **inArgs) -> None:
        self._BeginEpochTrain(inEpochIndex, **inArgs)
        for BatchIndex, (RealBatchData, _) in enumerate(inDataLoader):
            self._BatchTrain(BatchIndex, RealBatchData, **inArgs)
        self._EndEpochTrain(inEpochIndex, **inArgs)

    @abc.abstractmethod
    def _EndEpochTrain(self, inEpochIndex, **inArgs) -> None:
        pass

    @abc.abstractmethod
    def _BeginTrain(self, **inArgs) -> None:
        pass

    def Train(self, inNumEpochs : int, inDataLoader:DataLoader, **inArgs) -> None:
        self._CreateOptimizer()
        self._CreateLossFN()
        self._BeginTrain(**inArgs)
        for EpochIndex in range(inNumEpochs) :
            self._TrainEpoch(EpochIndex, inDataLoader, **inArgs)
        self._EndTrain(**inArgs)

    @abc.abstractmethod
    def _EndTrain(self, **inArgs) -> None:
        pass
