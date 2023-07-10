import sys
from torch.utils.data import DataLoader

from .CaseInsensitiveContainer import CaseInsensitiveDict, CaseInsensitiveList
import re

import keyboard

###################################################################################################

class Executor :
    def __init__(self, inModel) -> None:
        self.Model = inModel

        self.KVArgsForExec = CaseInsensitiveDict()
        self.ArgsForExec = CaseInsensitiveList()
        self.KVArgsForML = CaseInsensitiveDict()
        self.ArgsForML = CaseInsensitiveList()
        self.__GetArgs()
        
        self.bForceNewTrain     = False
        self.bForceIncTrain     = False
        self.bIncTrain          = False
        self.bForceEval         = False

        self.StartEpochIndex    = 0
        self.EpochIterCount     = 0
        self.__AnalyzeArgs()

###################################################################################################

    def Train(self, inDataLoader:DataLoader, *inArgsForML, **inKVArgsForML) :
        ## Only for Train
        keyboard.add_hotkey('ctrl + s', self.__HotKeySave)
        keyboard.add_hotkey('ctrl + x', self.__HotKeyExit)
        ##-----------------
        if self.bForceNewTrain or self.bIncTrain is False :
            self.Model.Train(inDataLoader, self.EpochIterCount, self.__CombineArgsForML(inArgsForML), self.__CombineKVArgsForML(inKVArgsForML))
        else :
            self.Model.IncTrain(inDataLoader, self.StartEpochIndex, self.EpochIterCount, self.__CombineArgsForML(inArgsForML), self.__CombineKVArgsForML(inKVArgsForML))
    
    ##----------------------------------------##
    
    def Eval(self, *inArgsForML, **inKVArgsForML) :
        return self.Model.Eval(self.StartEpochIndex, self.__CombineArgsForML(inArgsForML), self.__CombineKVArgsForML(inKVArgsForML))
    
    ##----------------------------------------##
    
    def Load(self, *inArgsForML, **inKVArgsForML) :
        return self.Model.LoadLastest(self.__CombineArgsForML(inArgsForML), self.__CombineKVArgsForML(inKVArgsForML))
    
    ##----------------------------------------##
    
    def IsExistModel(self) :
        return self.Model.IsExistModels()
    
    ##----------------------------------------##
    
    def ForceTrain(self) :
        return self.bForceNewTrain or self.bForceIncTrain

###################################################################################################

    def __GetArgs(self):
        for i in sys.argv :
            tmpi = i.casefold()
            if bool(re.match(r'^[-]{1,2}[\w]+=[\w]+', tmpi)):
                key, value = tmpi.split("=")
                if key.startswith("--"):
                    key = key.replace("--", "")
                    self.KVArgsForML[key]=value
                else:
                    key = key.replace("-", "")
                    self.KVArgsForExec[key]=value
            elif bool(re.match(r'^[-]{1,2}[\w]+', tmpi)):
                if tmpi.startswith("--"):
                    tmpi = tmpi.replace("--", "")
                    self.ArgsForML.append(tmpi)
                else:
                    tmpi = tmpi.replace("-", "")
                    self.ArgsForExec.append(tmpi)
            else : 
                self.ArgsForExec.append(tmpi)

    def __AnalyzeArgs(self):
        for CurrArg in self.ArgsForExec:
            if (CurrArg == "newtrain" or CurrArg == "new") :
                self.bForceNewTrain = True
            elif (CurrArg == "inctrain" or CurrArg == "inc"):
                self.bForceIncTrain = True
                self.bIncTrain = True
            elif (CurrArg == "eval"):
                self.bForceEval = True
            else:
                pass

        StartEpochIndex = self.KVArgsForExec.get("epoch")
        if StartEpochIndex is not None:
            self.StartEpochIndex = int(StartEpochIndex)
            self.bIncTrain = True

        EpochIterCount = self.KVArgsForExec.get("epochitercount")
        if EpochIterCount is not None:
            self.EpochIterCount =  int(EpochIterCount)

    def __CombineKVArgsForML(self, inKVArgs) :
        CombineDict = CaseInsensitiveDict(**inKVArgs)
        for key, value in self.KVArgsForML.items():
            CombineDict[key] = value

        return CombineDict
    
    def __CombineArgsForML(self, inArgs) :
        CombineList = CaseInsensitiveList(*inArgs)
        for value in self.ArgsForML:
            if value not in CombineList:
                CombineList.append(value)

        return CombineList
###################################################################################################

    def __HotKeySave(self):
        self.Model.ForceSaveAtEndEpoch()

    def __HotKeyExit(self):
        self.Model.ForceExitAtEndEpoch()

###################################################################################################
