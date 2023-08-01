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
            torch.save(Model.state_dict(), ModelFullPath)
            print("Save Model:" + ModelFullPath)

    def Load(self, inEpochIndex : int):
        for Name, _ in self.NNModuleDict.items():
            FilePath, FileName = self.GetFileFromValidLatestTimestampDirPath(Name, inEpochIndex)
            if FilePath is None:
                return False
            ModelFullPath = os.path.join(FilePath, FileName)
            self.NNModuleDict[Name].load_state_dict(torch.load(ModelFullPath))
            print("Load Model:" + ModelFullPath)

        return True

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
        self.NNModuleDict[inModelName].load_state_dict(torch.load(ModelFullPath))
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
            Model.load_state_dict(torch.load(ModelFullPath))
            print("Load Model:" + ModelFullPath)

############################################################################
