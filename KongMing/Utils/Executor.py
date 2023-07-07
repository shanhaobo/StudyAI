import sys
from torch.utils.data import DataLoader
from KongMing.Models.BaseModel import BaseModel

from .CaseInsensitiveContainer import CaseInsensitiveDict, CaseInsensitiveList
import re

class Executor :
    def __init__(self, inModel : BaseModel) -> None:
        self.Model = inModel

        self.ExecutorKVArgs = CaseInsensitiveDict()
        self.ExecutorArgs = CaseInsensitiveList()
        self.KVArgs = CaseInsensitiveDict()
        self.GetArgs()
        
        self.bForceNewTrain     = False
        self.bForceIncTrain     = False
        self.bIncTrain          = False
        self.bForceEval         = False

        self.StartEpochIndex    = 0
        self.EpochIterCount     = 0
        self.AnalyzeArgs()

    def Train(self, inDataLoader:DataLoader, *inArgs, **inKWArgs) :
        if self.bForceNewTrain or self.bIncTrain is False :
            self.Model.Train(inDataLoader, 0, self.EpochIterCount, CaseInsensitiveList(*inArgs), self.CombineKVArgs(**inKWArgs))
        else :
            self.Model.IncTrain(inDataLoader, self.StartEpochIndex, self.EpochIterCount, CaseInsensitiveList(*inArgs), self.CombineKVArgs(**inKWArgs))

    def Eval(self, *inArgs, **inKWArgs) :
        return self.Model.Eval(self.StartEpochIndex, CaseInsensitiveList(*inArgs), self.CombineKVArgs(**inKWArgs))

    def Load(self, *inArgs, **inKWArgs) :
        return self.Model.LoadLastest(CaseInsensitiveList(*inArgs), self.CombineKVArgs(**inKWArgs))
    
    def IsExistModel(self) :
        return self.Model.IsExistModels()
    
    def ForceTrain(self) :
        return self.bForceNewTrain or self.bForceIncTrain

###################################################################################################

    def GetArgs(self):
        for i in sys.argv :
            tmpi = i.casefold()
            Pattern = r'^[-]{1,2}[\w]+=[\w]+'
            if bool(re.match(Pattern, tmpi)):
                key, value = tmpi.split("=")
                if key.startswith("--"):
                    key = key.replace("--", "")
                    self.KVArgs[key]=value
                else:
                    key = key.replace("-", "")
                    self.ExecutorKVArgs[key]=value
            else:
                self.ExecutorArgs.append(tmpi)

    def AnalyzeArgs(self):
        for CurrArg in self.ExecutorArgs:
            if (CurrArg == "newtrain" or CurrArg == "new") :
                self.bForceNewTrain = True
            elif (CurrArg == "inctrain" or CurrArg == "inc"):
                self.bForceIncTrain = True
                self.bIncTrain = True
            elif (CurrArg == "eval"):
                self.bForceEval = True
            else:
                pass

        StartEpochIndex = self.ExecutorKVArgs.get("epoch")
        if StartEpochIndex is not None:
            self.StartEpochIndex = int(StartEpochIndex)
            self.bIncTrain = True

        EpochIterCount = self.ExecutorKVArgs.get("epochitercount")
        if EpochIterCount is not None:
            self.EpochIterCount =  int(EpochIterCount)


    def CombineKVArgs(self, **inKWArgs) :
        CombineDict = CaseInsensitiveDict(**self.KVArgs)
        for key, value in inKWArgs.items():
            kv = CombineDict.get(key)
            if kv is None:
                CombineDict[key] = value

        return CombineDict

###################################################################################################
