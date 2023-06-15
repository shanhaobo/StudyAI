from torch.utils.data import DataLoader

from Trainer.BaseTrainer import BaseTrainer
from Archiver.BaseArchiver import BaseArchiver

class BaseModel(object):
    def __init__(self, inTrainer : BaseTrainer, inArchiver : BaseArchiver):
        self.Trainer    = inTrainer
        self.Archiver   = inArchiver
        self.Device     = inTrainer.Device

        self.Trainer.BeginTrain.add(self.__BMBeginTrain)
        self.Trainer.EndBatchTrain.add(self.__BMEndBatchTrain)
        self.Trainer.EndEpochTrain.add(self.__BMEndEpochTrain)
        self.Trainer.EndTrain.add(self.__BMEndTrain)

    ###########################################################################################
    def Train(self, inDataLoader : DataLoader, inNumEpochs : int = 0, *inArgs, **inKWArgs) -> None:
        self.Trainer.Train(inDataLoader, inNumEpochs, 0, *inArgs, **inKWArgs)

    def IncTrain(self, inDataLoader : DataLoader, inNumEpochs : int = 0, *inArgs, **inKWArgs) -> None:
        EpochIndex = self.Archiver.LoadLastest(True)
        if (EpochIndex <= 0) : 
            return None
        self.Trainer.Train(inDataLoader, inNumEpochs, EpochIndex, *inArgs, **inKWArgs)

    def LoadLastest(self, *inArgs, **inKWArgs):
        EpochIndex = self.Archiver.LoadLastest(False)
        if (EpochIndex <= 0) :
            return False
        self.Trainer.CurrEpochIndex = EpochIndex
        return True
        
    def IsExistModels(self, inForTrain : bool = True, *inArgs, **inKWArgs) -> bool:
        return self.Archiver.IsExistModel(inForTrain, *inArgs, **inKWArgs)
    
    def Eval(self, *inArgs, **inKWArgs):
        self.LoadLastest(False)

    ###########################################################################################

    def __BMBeginTrain(self, *inArgs, **inKWArgs)->None:
        print("Begin Training...")

    ############################################

    def __BMEndBatchTrain(self, *inArgs, **inKWArgs) -> None:
        print("End Batch Train")

    def __BMEndEpochTrain(self, *inArgs, **inKWArgs) -> None:
        interval = inKWArgs["SaveModelInterval"]
        if (self.Trainer.CurrEpochIndex % interval == 0) and (self.Trainer.CurrEpochIndex > 0) :
            print("Epoch:{} Save Models".format(self.Trainer.CurrEpochIndex + 1))
            self.Archiver.Save(self.Trainer.CurrEpochIndex + 1)

    def __BMEndTrain(self, *inArgs, **inKWArgs)->None:
        self.Archiver.Save(self.Trainer.CurrEpochIndex + 1)
        print("End Train")

    ###########################################################################################
