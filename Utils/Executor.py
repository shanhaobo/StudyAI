import sys

from Models.BaseModel import BaseModel

class Executor :
    def __init__(self, inModel : BaseModel) -> None:
        self.Model = inModel

    def Train(self, bContinue:bool) :
        pass

    def Eval(self) :
        pass

    def Run(self) :
        for i in sys.argv :
            if (i == "train") :
                self.Train(True)
            elif (i == "trainnew") :
                self.Train(False)
            elif (i == "eval") :
                self.Eval()
            else :
                self.Train(False)
