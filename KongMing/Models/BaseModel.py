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
    def Train(self, inDataLoader : DataLoader, inStartEpochNum : int = 0, inEpochIterCount : int = 0, *inArgs, **inKWArgs) -> None:
        self.Trainer.Train(inDataLoader, inStartEpochNum, inEpochIterCount, *inArgs, **inKWArgs)

    def IncTrain(self, inDataLoader : DataLoader, inStartEpochNum : int = 0, inEpochIterCount : int = 0, *inArgs, **inKWArgs) -> None:
        if inStartEpochNum > 0 and self.Archiver.Load(inStartEpochNum):
            pass
        else:
            inStartEpochNum = self.Archiver.LoadLastest()
            if (inStartEpochNum < 0) : 
                inStartEpochNum = 0

        self.Trainer.Train(inDataLoader, inStartEpochNum, inEpochIterCount, *inArgs, **inKWArgs)

    def LoadLastest(self, *inArgs, **inKWArgs):
        EpochIndex = self.Archiver.LoadLastest()
        if (EpochIndex <= 0) :
            return False
        self.Trainer.CurrEpochIndex = EpochIndex
        return True
    
    def Load(self, inEpoch, *inArgs, **inKWArgs):
        self.Archiver.Load(inEpoch)
        
    def IsExistModels(self) -> bool:
        return self.Archiver.IsExistModel()
    
    def Eval(self, inEpoch = -1, *inArgs, **inKWArgs):
        self.Archiver.Eval()

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
