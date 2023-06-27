import abc
import torch

from torch import Tensor
from torch.optim.optimizer import Optimizer

from torch.utils.data import DataLoader

from KongMing.Utils.Delegate import Delegate

class BaseTrainer(abc.ABC):
    def __init__(self, inLearningRate) -> None:
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


    def __DontOverride__Train(self, inDataLoader:DataLoader, inNumEpochs : int = 0, inStartEpochIndex : int = 0, *inArgs, **inKWArgs) -> None:
        # Begin Train
        # Create Optimizer & Loss Function
        self._CreateOptimizer()
        self._CreateLossFN()
        self.BeginTrain(*inArgs, **inKWArgs)

        if inNumEpochs <= 0 :
            self.CurrEpochIndex = inStartEpochIndex
            while True:
                self.__DontOverride__EpochTrain(inDataLoader, *inArgs, **inKWArgs)
                self.CurrEpochIndex += 1
        else:
            # For Each Epoch Train
            for self.CurrEpochIndex in range(inStartEpochIndex, inNumEpochs) :
                self.__DontOverride__EpochTrain(inDataLoader, *inArgs, **inKWArgs)
        
        # End Train
        self.EndTrain(*inArgs, **inKWArgs)


    def Train(self, inDataLoader : DataLoader, inNumEpochs : int = 0, inStartEpochIndex : int = 0, *inArgs, **inKWArgs) -> None:
        self.__DontOverride__Train(inDataLoader, inNumEpochs, inStartEpochIndex, *inArgs, **inKWArgs)
