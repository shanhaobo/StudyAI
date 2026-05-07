import sys
from torch.utils.data import DataLoader

from .CaseInsensitiveContainer import CaseInsensitiveDict, CaseInsensitiveList
import re

import keyboard

###################################################################################################

def ResolveModelTag() -> str :
    """轻量预解析：在 Factory 构造之前从 sys.argv 抓 --ModelTag= 的值。
    Executor 的完整解析依赖 self，但 Archiver 的根路径要在构造时就拼好；
    把这一小段抽出来，避免重复全 argv 扫描或循环依赖。"""
    for raw in sys.argv:
        if not raw.startswith("--"):
            continue
        if "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        if key[2:].casefold() == "modeltag" and value:
            return value
    return None

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

    def Train(self, inDataLoader:DataLoader, *inArgsForML, inValidLoader : DataLoader = None, **inKVArgsForML) :
        ## Only for Train —— 全局热键依赖 keyboard 包；
        ## Linux 非 root / Mac / 容器里注册会抛权限错误，但训练本身不该被它阻塞。
        try:
            keyboard.add_hotkey('ctrl + s', self.__HotKeySave)
            keyboard.add_hotkey('ctrl + x', self.__HotKeyExit)
        except Exception as e:
            print("[Executor] Hotkey unavailable, training continues without Ctrl+S/Ctrl+X. Reason:", e)
        ##-----------------
        if self.bForceNewTrain or self.bIncTrain is False :
            self.Model.NewTrain(inDataLoader, self.EpochIterCount, self.__CombineArgsForML(inArgsForML), self.__CombineKVArgsForML(inKVArgsForML), inValidLoader=inValidLoader)
        else :
            self.Model.IncTrain(inDataLoader, self.StartEpochIndex, self.EpochIterCount, self.__CombineArgsForML(inArgsForML), self.__CombineKVArgsForML(inKVArgsForML), inValidLoader=inValidLoader)
    
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
        return (self.bForceNewTrain or self.bForceIncTrain) and self.bForceEval is False

    ##----------------------------------------##
    
    def IsNewTrain(self) : 
        return (self.bForceNewTrain or self.bIncTrain is False) and self.bForceEval is False

    ##----------------------------------------##
    
    def IsEval(self) :
        return self.bForceEval or (self.bForceNewTrain is False and self.bForceIncTrain is False)


###################################################################################################

    def __GetArgs(self):
        for raw in sys.argv :
            # 只对 key 部分 casefold；value 保留原文（路径/tag/字符串规格不能丢大小写）
            if bool(re.match(r'^[-]{1,2}[\w]+=.+', raw)):
                key, _, value = raw.partition("=")
                if key.startswith("--"):
                    key = key[2:].casefold()
                    self.KVArgsForML[key]=value
                else:
                    key = key.lstrip("-").casefold()
                    self.KVArgsForExec[key]=value
            elif bool(re.match(r'^[-]{1,2}[\w]+', raw)):
                tmpi = raw.casefold()
                if tmpi.startswith("--"):
                    tmpi = tmpi[2:]
                    self.ArgsForML.append(tmpi)
                else:
                    tmpi = tmpi.lstrip("-")
                    self.ArgsForExec.append(tmpi)
            else :
                self.ArgsForExec.append(raw.casefold())

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
