from torch.utils.data import DataLoader

from KongMing.Trainer.BaseTrainer import BaseTrainer
from KongMing.Archiver.BaseArchiver import BaseArchiver

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
        if inNumEpochs > 0:
            if self.Archiver.Load(inNumEpochs, True):
                self.Trainer.Train(inDataLoader, inNumEpochs + 1, *inArgs, **inKWArgs)
        else:
            EpochIndex = self.Archiver.LoadLastest(True)
            if (EpochIndex < 0) : 
                EpochIndex = 0
            else:
                EpochIndex += 1
            self.Trainer.Train(inDataLoader, EpochIndex, *inArgs, **inKWArgs)

    def LoadLastest(self, *inArgs, **inKWArgs):
        EpochIndex = self.Archiver.LoadLastest(False)
        if (EpochIndex <= 0) :
            return False
        self.Trainer.CurrEpochIndex = EpochIndex
        return True
    
    def Load(self, inEpoch, *inArgs, **inKWArgs):
        self.Archiver.Load(inEpoch, False)
        
    def IsExistModels(self, inForTrain : bool = True, *inArgs, **inKWArgs) -> bool:
        return self.Archiver.IsExistModel(inForTrain, *inArgs, **inKWArgs)
    
    def Eval(self, inEpoch = -1, *inArgs, **inKWArgs):
        if inEpoch < 0:
            self.LoadLastest(*inArgs, **inKWArgs)
        else:
            self.Load(inEpoch)


    ###########################################################################################

    def _SumParameters(self,inNN):
        return sum(p.nelement() for p in inNN.parameters())

    ###########################################################################################

    def __BMBeginTrain(self, *inArgs, **inKWArgs)->None:
        print("Begin Training...")

    ############################################

    def __BMEndBatchTrain(self, *inArgs, **inKWArgs) -> None:
        pass

    def __BMEndEpochTrain(self, *inArgs, **inKWArgs) -> None:
        interval = inKWArgs["SaveModelInterval"]
        if (self.Trainer.CurrEpochIndex + 1) % interval == 0:
            print("Epoch:{} Save Models".format(self.Trainer.CurrEpochIndex))
            self.Archiver.Save(self.Trainer.CurrEpochIndex)

    def __BMEndTrain(self, *inArgs, **inKWArgs)->None:
        self.Archiver.Save(self.Trainer.CurrEpochIndex)
        print("End Train")

    ###########################################################################################
