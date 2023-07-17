from torch.utils.data import DataLoader

from KongMing.Utils.CaseInsensitiveContainer import CaseInsensitiveList, CaseInsensitiveDict

from KongMing.Trainer.BaseTrainer import BaseTrainer
from KongMing.Archiver.BaseArchiver import BaseArchiver

class BaseModelFactory(object):
    def __init__(self, inTrainer : BaseTrainer, inArchiver : BaseArchiver):
        self.Trainer        = inTrainer
        self.Archiver       = inArchiver
        self.Device         = inTrainer.Device

        self.Trainer.BeginTrain.add(self.__BMBeginTrain)
        self.Trainer.EndBatchTrain.add(self.__BMEndBatchTrain)
        self.Trainer.EndEpochTrain.add(self.__BMEndEpochTrain)
        self.Trainer.EndTrain.add(self.__BMEndTrain)

        self.ForceSave      = False

        self.SaveInterval   = 10

    ###########################################################################################

    def Train(self, inDataLoader : DataLoader, inEpochIterCount : int, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None) -> None:
        self.Trainer.Train(inDataLoader, 0, inEpochIterCount, inArgs, inKVArgs)

    def IncTrain(self, inDataLoader : DataLoader, inStartEpochNum : int, inEpochIterCount : int, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None) -> None:
        if inStartEpochNum >= 0 and self.Archiver.Load(inStartEpochNum):
            pass
        else:
            inStartEpochNum = self.Archiver.LoadLastest()
        self.Trainer.Train(inDataLoader, inStartEpochNum + 1, inEpochIterCount, inArgs, inKVArgs)

    def LoadLastest(self, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None):
        EpochIndex = self.Archiver.LoadLastest()
        if (EpochIndex is None) or (EpochIndex < 0):
            return False
        self.Trainer.CurrEpochIndex = EpochIndex + 1
        return True
    
    def Load(self, inEpoch, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None):
        self.Archiver.Load(inEpoch)
        
    def IsExistModels(self) -> bool:
        return self.Archiver.IsExistModel()
    
    def Eval(self, inEpoch, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None):
        self.Archiver.Eval()
        if inEpoch <= 0:
            self.LoadLastest(inArgs, inKVArgs)
        else:
            self.Load(inEpoch)

    ###########################################################################################

    def _SumParameters(self,inNN):
        return sum(p.nelement() for p in inNN.parameters())

    ###########################################################################################

    def __BMBeginTrain(self, inArgs, inKVArgs)->None:
        print("Begin Training...")
        SaveInterval = inKVArgs.get("SaveInterval")
        if SaveInterval is not None:
            self.SaveInterval = int(SaveInterval)

    ############################################

    def __BMEndBatchTrain(self, inArgs, inKVArgs) -> None:
        pass

    def __BMEndEpochTrain(self, inArgs, inKVArgs) -> None:
        if self.ForceSave or ((self.Trainer.CurrEpochIndex + 1) % self.SaveInterval == 0):
            if self.ForceSave :
                self.ForceSave = False
                print("Epoch:{} Force Save Models".format(self.Trainer.CurrEpochIndex))
            else:
                print("Epoch:{} Save Models".format(self.Trainer.CurrEpochIndex))
            self.Archiver.Save(self.Trainer.CurrEpochIndex)

    def __BMEndTrain(self, inArgs, inKVArgs)->None:
        self.Archiver.Save(self.Trainer.CurrEpochIndex)
        print("End Train!!!")

    ###########################################################################################

    def ForceSaveAtEndEpoch(self) -> None:
        print("Accept Force Save.............")
        self.ForceSave = True

    def ForceExitAtEndEpoch(self) -> None:
        print("Accept Soft Exit.............")
        self.Trainer.SoftExit = True

    ###########################################################################################
