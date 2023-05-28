import abc
import torch

from torch import Tensor
from torch.optim.optimizer import Optimizer

from torch.utils.data import DataLoader

from Utils.Delegate import Delegate

class BaseTrainer(abc.ABC):
    def __init__(self, inLearningRate) -> None:
        self.Device             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


    def __DontOverride__Train(self, inNumEpochs : int, inDataLoader:DataLoader, *inArgs, **inKWArgs) -> None:
        # Begin Train
        # Create Optimizer & Loss Function
        self._CreateOptimizer()
        self._CreateLossFN()
        self.BeginTrain(*inArgs, **inKWArgs)

        # For Each Epoch Train
        for self.CurrEpochIndex in range(inNumEpochs) :
            self.__DontOverride__EpochTrain(inDataLoader, *inArgs, **inKWArgs)
        
        # End Train
        self.EndTrain(*inArgs, **inKWArgs)


    def Train(self, inNumEpochs : int, inDataLoader:DataLoader, *inArgs, **inKWArgs) -> None:
        self.__DontOverride__Train(inNumEpochs, inDataLoader, *inArgs, **inKWArgs)
