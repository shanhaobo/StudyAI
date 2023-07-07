from torch.utils.data import DataLoader

import keyboard

from KongMing.Utils.CaseInsensitiveContainer import CaseInsensitiveList, CaseInsensitiveDict

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

        self.CurrEpochForceSave           = False
        keyboard.add_hotkey('ctrl + s', self.__CurrEpochForceSave)

        self.SaveInterval = 10

    ###########################################################################################
    def Train(self, inDataLoader : DataLoader, inStartEpochNum : int, inEpochIterCount : int, inArgs : CaseInsensitiveList = None, inKWArgs : CaseInsensitiveDict = None) -> None:
        self.Trainer.Train(inDataLoader, inStartEpochNum, inEpochIterCount, inArgs, inKWArgs)

    def IncTrain(self, inDataLoader : DataLoader, inStartEpochNum : int, inEpochIterCount : int, inArgs : CaseInsensitiveList = None, inKWArgs : CaseInsensitiveDict = None) -> None:
        if inStartEpochNum > 0 and self.Archiver.Load(inStartEpochNum):
            pass
        else:
            inStartEpochNum = self.Archiver.LoadLastest()
            if (inStartEpochNum < 0) : 
                inStartEpochNum = 0

        self.Trainer.Train(inDataLoader, inStartEpochNum, inEpochIterCount, inArgs, inKWArgs)

    def LoadLastest(self, inArgs : CaseInsensitiveList = None, inKWArgs : CaseInsensitiveDict = None):
        EpochIndex = self.Archiver.LoadLastest()
        if (EpochIndex <= 0) :
            return False
        self.Trainer.CurrEpochIndex = EpochIndex
        return True
    
    def Load(self, inEpoch, inArgs : CaseInsensitiveList = None, inKWArgs : CaseInsensitiveDict = None):
        self.Archiver.Load(inEpoch)
        
    def IsExistModels(self) -> bool:
        return self.Archiver.IsExistModel()
    
    def Eval(self, inEpoch = -1, inArgs : CaseInsensitiveList = None, inKWArgs : CaseInsensitiveDict = None):
        self.Archiver.Eval()

        if inEpoch <= 0:
            self.LoadLastest(inArgs, inKWArgs)
        else:
            self.Load(inEpoch)


    ###########################################################################################

    def _SumParameters(self,inNN):
        return sum(p.nelement() for p in inNN.parameters())

    ###########################################################################################

    def __BMBeginTrain(self, *inArgs, **inKWArgs)->None:
        print("Begin Training...")
        SaveInterval = inKWArgs.get("saveinterval")
        if SaveInterval is not None:
            self.SaveInterval = int(SaveInterval)

    ############################################

    def __BMEndBatchTrain(self, *inArgs, **inKWArgs) -> None:
        pass

    def __BMEndEpochTrain(self, *inArgs, **inKWArgs) -> None:
        if self.CurrEpochForceSave or ((self.Trainer.CurrEpochIndex + 1) % self.SaveInterval == 0):
            self.CurrEpochForceSave = False
            print("Epoch:{} Save Models".format(self.Trainer.CurrEpochIndex))
            self.Archiver.Save(self.Trainer.CurrEpochIndex)

    def __BMEndTrain(self, *inArgs, **inKWArgs)->None:
        self.Archiver.Save(self.Trainer.CurrEpochIndex)
        print("End Train")

    ###########################################################################################

    def __CurrEpochForceSave(self) -> None:
        print("Force Saveing.............")
        self.CurrEpochForceSave = True
