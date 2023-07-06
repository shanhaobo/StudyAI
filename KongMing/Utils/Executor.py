import sys
from torch.utils.data import DataLoader
from KongMing.Models.BaseModel import BaseModel

import re

class Executor :
    def __init__(self, inModel : BaseModel) -> None:
        self.Model = inModel
        self.Args = {}

        self.GetArgs()

    def Train(self, inDataLoader:DataLoader, *inArgs, **inKWArgs) :
        bIncTrain = False
        bForceNewTrain = False
        for i in sys.argv :
            tmpi = i.casefold()
            if (tmpi == "inctrain" or tmpi == "inc") :
                bIncTrain = True
            elif (tmpi == "newtrain" or tmpi == "new") :
                bIncTrain = False
                bForceNewTrain = True
        
        if bForceNewTrain is False:
            StartEpochIndex = self.Args.get("epoch")
            if StartEpochIndex is not None:
                StartEpochIndex = int(StartEpochIndex)
                bIncTrain = True

        EpochIterCount = 0
        EICformArgs = self.Args.get("epochitercount")
        if EICformArgs is not None:
            EpochIterCount =  int(EICformArgs)

        if bForceNewTrain or bIncTrain is False :
            self.Model.Train(inDataLoader, 0, EpochIterCount, *inArgs, **inKWArgs)
        else :
            if StartEpochIndex is None:
                StartEpochIndex = -1
            self.Model.IncTrain(inDataLoader, StartEpochIndex, EpochIterCount, *inArgs, **inKWArgs)

    def Eval(self, *inArgs, **inKWArgs) :
        Epoch = self.Args.get("epoch")
        if Epoch is None:
            Epoch = -1
        else:
            Epoch = int(Epoch)

        return self.Model.Eval(Epoch, *inArgs, **inKWArgs)

    def GetArgs(self):
        for i in sys.argv :
            tmpi = i.casefold()
            pattern = r'^-[\w]+=[\w]+'
            if bool(re.match(pattern, tmpi)):
                key, value = tmpi.split("=")
                key = key.replace("-", "")
                self.Args[key]=value

    def Load(self, *inArgs, **inKWArgs) :
        if self.Model.LoadLastest(*inArgs, **inKWArgs) :
            return True
        return False
    
    def IsExistModel(self) :
        return self.Model.IsExistModels()
    
    def ReadyTrain(self) :
        for i in sys.argv :
            tmpi = i.casefold()
            if (i == "inctrain" or i == "inc") :
                return True
            if (i == "train") :
                return True
            if (i == "newtrain" or i == "new") :
                return True
        return False
