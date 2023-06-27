import sys
from torch.utils.data import DataLoader
from Models.BaseModel import BaseModel

class Executor :
    def __init__(self, inModel : BaseModel) -> None:
        self.Model = inModel

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
        return self.Model.Eval(*inArgs, **inKWArgs)

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
