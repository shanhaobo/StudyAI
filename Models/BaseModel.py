from torch.utils.data import DataLoader

from Trainer.BaseTrainer import BaseTrainer
from Utils.Archiver import BaseArchiver

class BaseModel(object):
    def __init__(self, inTrainer : BaseTrainer, inArchiver : BaseArchiver):
        self.Trainer    = inTrainer
        self.Archiver   = inArchiver

    def Train(self, inDataLoader : DataLoader, inNumEpochs : int = 0, *inArgs, **inKWArgs) -> None:
        self.Trainer.Train(inDataLoader, inNumEpochs, 0, *inArgs, **inKWArgs)

    def IncTrain(self, inDataLoader : DataLoader, inNumEpochs : int = 0, *inArgs, **inKWArgs) -> None:
        self.Archiver.LoadLastest(True)
        self.Trainer.Train(inDataLoader, inNumEpochs, 0, *inArgs, **inKWArgs)

    def LoadLastest(self, *inArgs, **inKWArgs):
        bSuccess, EpochIndex = self.Archiver.LoadLastest(False)
        if bSuccess == False :
            return False
        self.Trainer.CurrEpochIndex = EpochIndex
        return True
        
    def IsExistModels(self, inForTrain : bool = True, *inArgs, **inKWArgs) -> bool:
        return self.Archiver.IsExistModel(inForTrain, *inArgs, **inKWArgs)
    
