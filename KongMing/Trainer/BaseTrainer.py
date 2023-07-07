import abc
import torch

from torch import Tensor
from torch.optim.optimizer import Optimizer

from torch.utils.data import DataLoader

from KongMing.Utils.Delegate import Delegate

class BaseTrainer(abc.ABC):
    def __init__(self, inLearningRate, inLogRootPath) -> None:
        self.Device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.get_device_name(self.Device.index))
        self.LearningRate       = inLearningRate

        self.BeginTrain         = Delegate()
        self.EndTrain           = Delegate()

        self.BeginEpochTrain    = Delegate()
        self.EndEpochTrain      = Delegate()

        self.BeginBatchTrain    = Delegate()
        self.EndBatchTrain      = Delegate()

        self.CurrEpochIndex     = 0
        self.CurrBatchIndex     = 0

        self.EndEpochIndex      = 0

        self.SoftExit           = False

        self.LogRootPath        = inLogRootPath

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
    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :
        pass
    
    def __DontOverride__EpochTrain(self, inDataLoader:DataLoader, inArgs, inKVArgs) -> None:
        # Begin Epoch Train 
        # call BeginEpochTrain
        self.BeginEpochTrain(inArgs, inKVArgs)

        # For Each Batch Train
        for self.CurrBatchIndex, (CurrBatchData, CurrBatchLabel) in enumerate(inDataLoader):
            self.BeginBatchTrain(inArgs, inKVArgs)
            self._BatchTrain(CurrBatchData, CurrBatchLabel, inArgs, inKVArgs)
            self.EndBatchTrain(inArgs, inKVArgs)
        
        # End Epoch Train
        # call EndEpochTrain
        self.EndEpochTrain(inArgs, inKVArgs)


    def __DontOverride__Train(self, inDataLoader:DataLoader, inStartEpochIndex : int, inEpochIterCount : int, inArgs, inKVArgs) -> None:
        # Begin Train
        # Create Optimizer & Loss Function
        self._CreateOptimizer()
        self._CreateLossFN()
        self.BeginTrain(inArgs, inKVArgs)

        self.CurrEpochIndex = inStartEpochIndex
        self.EndEpochIndex = (self.CurrEpochIndex + inEpochIterCount) if (inEpochIterCount > 0) else 0
        while self.__Continue_EpochIterCount():
            self.__DontOverride__EpochTrain(inDataLoader, inArgs, inKVArgs)
            if self.SoftExit or self._CheckEndEpoch():
                break
            self.CurrEpochIndex += 1

        # End Train
        self.EndTrain(inArgs, inKVArgs)

    def Train(self, inDataLoader : DataLoader, inStartEpochIndex : int, inEpochIterCount : int, inArgs, inKVArgs) -> None:
        if inStartEpochIndex < 0:
            inStartEpochIndex = 0
        self.__DontOverride__Train(inDataLoader, inStartEpochIndex, inEpochIterCount, inArgs, inKVArgs)

    def _CheckEndEpoch(self)->bool:
        return False
    
    def __Continue_EpochIterCount(self) -> bool:
        if self.EndEpochIndex <= 0:
            return True
        
        return self.CurrEpochIndex < self.EndEpochIndex
