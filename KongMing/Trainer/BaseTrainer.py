import abc
import torch

import keyboard

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

        self.SoftExit           = False
        keyboard.add_hotkey('ctrl + x', self.__SoftExit)

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
    def _BatchTrain(self, inBatchData, inBatchLabel, *inArgs, **inKWArgs) :
        pass
    
    def __DontOverride__EpochTrain(self, inDataLoader:DataLoader, *inArgs, **inKWArgs) -> None:
        # Begin Epoch Train 
        # call BeginEpochTrain
        self.BeginEpochTrain(*inArgs, **inKWArgs)

        # For Each Batch Train
        for self.CurrBatchIndex, (CurrBatchData, CurrBatchLabel) in enumerate(inDataLoader):
            self.BeginBatchTrain(*inArgs, **inKWArgs)
            self._BatchTrain(CurrBatchData, CurrBatchLabel, *inArgs, **inKWArgs)
            self.EndBatchTrain(*inArgs, **inKWArgs)
        
        # End Epoch Train
        # call EndEpochTrain
        self.EndEpochTrain(*inArgs, **inKWArgs)


    def __DontOverride__Train(self, inDataLoader:DataLoader, inStartEpochIndex : int = 0, *inArgs, **inKWArgs) -> None:
        # Begin Train
        # Create Optimizer & Loss Function
        self._CreateOptimizer()
        self._CreateLossFN()
        self.BeginTrain(*inArgs, **inKWArgs)

        self.CurrEpochIndex = inStartEpochIndex
        while self._Continue():
            self.__DontOverride__EpochTrain(inDataLoader, *inArgs, **inKWArgs)
            self.CurrEpochIndex += 1
            if self.SoftExit:
                break
        
        # End Train
        self.EndTrain(*inArgs, **inKWArgs)

    def Train(self, inDataLoader : DataLoader, inStartEpochIndex : int = 0, *inArgs, **inKWArgs) -> None:
        self.__DontOverride__Train(inDataLoader, inStartEpochIndex, *inArgs, **inKWArgs)

    def _Continue(self)->bool:
        return True

    def __SoftExit(self):
        print("Soft Exiting......................")
        self.SoftExit = True
