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
        for i in sys.argv :
            tmpi = i.casefold()
            if (tmpi == "inctrain" or tmpi == "inc") :
                bIncTrain = True
            elif (tmpi == "newtrain" or tmpi == "new") :
                bIncTrain = False
        
        if bIncTrain :
            self.Model.IncTrain(inDataLoader, 0, *inArgs, **inKWArgs)
        else :
            self.Model.Train(inDataLoader, 0, *inArgs, **inKWArgs)

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
        return self.Model.IsExistModels(True)
    
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
