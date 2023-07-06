import sys
from torch.utils.data import DataLoader
from KongMing.Models.BaseModel import BaseModel

import re

class Executor :
    def __init__(self, inModel : BaseModel) -> None:
        self.Model = inModel

        self.KVArgs = {}
        self.Args = []
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
            self.Model.Train(inDataLoader, 0, self.EpochIterCount, *inArgs, **inKWArgs)
        else :
            self.Model.IncTrain(inDataLoader, self.StartEpochIndex, self.EpochIterCount, *inArgs, **inKWArgs)

    def Eval(self, *inArgs, **inKWArgs) :
        return self.Model.Eval(self.StartEpochIndex, *inArgs, **inKWArgs)

    def Load(self, *inArgs, **inKWArgs) :
        return self.Model.LoadLastest(*inArgs, **inKWArgs)
    
    def IsExistModel(self) :
        return self.Model.IsExistModels()
    
    def ForceTrain(self) :
        return self.bForceNewTrain or self.bForceIncTrain

###################################################################################################

    def GetArgs(self):
        for i in sys.argv :
            tmpi = i.casefold()
            pattern = r'^-[\w]+=[\w]+'
            if bool(re.match(pattern, tmpi)):
                key, value = tmpi.split("=")
                key = key.replace("-", "")
                self.KVArgs[key]=value
            else:
                self.Args.append(tmpi)

    def AnalyzeArgs(self):
        for CurrArg in self.Args:
            if (CurrArg == "newtrain" or CurrArg == "new") :
                self.bForceNewTrain = True
            elif (CurrArg == "inctrain" or CurrArg == "inc"):
                self.bForceIncTrain = True
                self.bIncTrain = True
            elif (CurrArg == "eval"):
                self.bForceEval = True
            else:
                pass

        StartEpochIndex = self.KVArgs.get("epoch")
        if StartEpochIndex is not None:
            self.StartEpochIndex = int(StartEpochIndex)
            self.bIncTrain = True

        EpochIterCount = self.KVArgs.get("epochitercount")
        if EpochIterCount is not None:
            self.EpochIterCount =  int(EpochIterCount)


###################################################################################################
