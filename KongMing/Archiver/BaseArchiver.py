import os
import torch
from KongMing.Utils.ModelFileOp import FindFileWithMaxNum

from .Path.FileManagerWithNum import FileManagerWithNum

from typing import Dict as TypedDict
from typing import List as TypedList

class BaseArchiver(object):
    def __init__(self, inModelRootFolderPath : str, inNNModuleNameOnlyForTrain : TypedList[str] = []) -> None:
        self.ModelArchiveRootFolderPath = os.path.join(inModelRootFolderPath, "ArchivedModels")

        self.FileNameManager            = FileManagerWithNum(self.ModelArchiveRootFolderPath, ".pkl", 100, True)

        self.SaveEpochIndex             = -1
        self.NNModuleDict : TypedDict[str, torch.nn.Module] = {}
        self.NNModuleNameOnlyForTrain   = inNNModuleNameOnlyForTrain

############################################################################
    def GetCurrTrainRootPath(self):
        return self.FileNameManager.MakeAndGetRootPath()
############################################################################

    def IsExistModel(self) -> bool:
        for Name, _ in self.NNModuleDict.items():
            Path, _ = self.FindLatestModelFile(Name)
            if Path is None:
                return False

        return True

############################################################################

    def Eval(self):
        for Name in self.NNModuleNameOnlyForTrain:
            del self.NNModuleDict[Name]

############################################################################

    def MakeNeuralNetworkArchiveFullPath(self, inNeuralNetworkName : str, inEpochIndex : int) -> str:
        return self.FileNameManager.MakeFileFullPathAndFileName(FileName = inNeuralNetworkName, Num = inEpochIndex)

    def GetFileFromValidLatestTimestampDirPath(self, inNeuralNetworkName : str, inEpochIndex : int) -> str:
        return self.FileNameManager.GetFilePathAndNameFromTimestampDirPathByEpoch(FileName = inNeuralNetworkName, Num = inEpochIndex)

    def GetLatestModelFolder(self) -> str :
        _, LatestLeafFolderPath, _ = self.FileNameManager.GetValidLatestTimestampDirInfo()

        return LatestLeafFolderPath

    def FindLatestModelFile(self, inModelName : str):
        LatestFolderPath = self.GetLatestModelFolder()
        if LatestFolderPath is None :
            return None, None

         # 返回数字最大（也就是最新）的文件
        FileName, MaxNum =  FindFileWithMaxNum(os.listdir(LatestFolderPath), inModelName, "*", "pkl")
        if FileName is None :
            return None, None

        return os.path.join(LatestFolderPath, FileName), MaxNum

############################################################################

    def Save(self, inEpochIndex : int) -> None:
        # if SaveEpochIndex == inEpochIndex means already saved
        if (self.SaveEpochIndex < inEpochIndex):
            self._Save(inEpochIndex=inEpochIndex)
            self.SaveEpochIndex = inEpochIndex

    def _Save(self, inEpochIndex : int) -> None:
        for Name, Model in self.NNModuleDict.items():
            ModelFolderPath, ModelFileName = self.MakeNeuralNetworkArchiveFullPath(Name, inEpochIndex)
            os.makedirs(ModelFolderPath, exist_ok=True)
            ModelFullPath = os.path.join(ModelFolderPath, ModelFileName)
            # BaseNNModel 走 archive 协议（带 Optimizer / LRScheduler 状态）；
            # 其它普通 nn.Module 退回旧的 state_dict。
            if hasattr(Model, "StateDictForArchive"):
                torch.save(Model.StateDictForArchive(), ModelFullPath)
            else:
                torch.save(Model.state_dict(), ModelFullPath)
            print("Save Model:" + ModelFullPath)

    def Load(self, inEpochIndex : int):
        for Name, _ in self.NNModuleDict.items():
            FilePath, FileName = self.GetFileFromValidLatestTimestampDirPath(Name, inEpochIndex)
            if FilePath is None:
                return False
            ModelFullPath = os.path.join(FilePath, FileName)
            self.__LoadInto(self.NNModuleDict[Name], ModelFullPath)
            print("Load Model:" + ModelFullPath)

        return True

    @staticmethod
    def __LoadInto(inModule : torch.nn.Module, inFullPath : str) -> None:
        # 兼容新旧两种 checkpoint：BaseNNModel 走 archive 协议，普通 nn.Module 走 state_dict。
        # weights_only=True 是 PyTorch 2.6+ 的默认值，显式写出来防止 unpickle 任意类——
        # 我们存的内容只有 dict / OrderedDict / Tensor / Python 标量，纯白名单类型，没问题。
        Loaded = torch.load(inFullPath, weights_only=True)
        if hasattr(inModule, "LoadStateDictFromArchive"):
            inModule.LoadStateDictFromArchive(Loaded)
        else:
            # 旧 .pkl 是 state_dict；新格式万一被普通 nn.Module 撞上，取 "Model" 子项
            if isinstance(Loaded, dict) and ("Model" in Loaded):
                inModule.load_state_dict(Loaded["Model"])
            else:
                inModule.load_state_dict(Loaded)

    def LoadLastest(self):
        MaxEpochIndex = -1
        for Name, _ in self.NNModuleDict.items():
            EpochIndex = self.LoadLastestByModelName(Name)
            if EpochIndex is None:
                return None
            if EpochIndex > MaxEpochIndex :
                MaxEpochIndex = EpochIndex
        return MaxEpochIndex

    def LoadLastestByModelName(self, inModelName : str):
        ModelFullPath, EpochIndex = self.FindLatestModelFile(inModelName)
        if ModelFullPath is None :
            return None
        self.__LoadInto(self.NNModuleDict[inModelName], ModelFullPath)
        print("Load Model:" + ModelFullPath)
        return EpochIndex

    def LoadModelByTimestamp(self, inTimestamp:str, inEpochIndex):
        for Name, Model in self.NNModuleDict.items():
            ModelFullPath = self.FileNameManager.GetFilePathByTimestamp(
                inTimestamp=inTimestamp,
                Num=inEpochIndex,
                FileName=Name
            )
            if ModelFullPath is None :
                return None
            self.__LoadInto(Model, ModelFullPath)
            print("Load Model:" + ModelFullPath)

############################################################################
