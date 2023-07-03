import os
import torch
from datetime import datetime
from KongMing.Utils.ModelFileOp import FindFileWithMaxNum

from .Path.FileManagerWithNum import FileManagerWithNum

class BaseArchiver(object):
    def __init__(self, inModelPrefix : str, inModelRootFolderPath : str = ".") -> None:
        self.ModelPrefix                = inModelPrefix
        self.ModelRootFolderPath        = inModelRootFolderPath
        self.ModelArchiveRootFolderPath = os.path.join(self.ModelRootFolderPath, self.ModelPrefix)
        self.ModelArchiveFolderPath     = self.ModelArchiveRootFolderPath

        self.FileNameManager = FileManagerWithNum(self.ModelArchiveRootFolderPath, ".pkl", 100, True)

        self.SaveEpochIndex             = -1
        self.NNModelDict                = {}

############################################################################
    def GetCurrTrainRootPath(self):
        return self.FileNameManager.GetRootPath()
############################################################################

    def IsExistModel(self, inForTrain : bool = True) -> bool:
        for Name, Model in self.NNModelDict.items():
            Path, _ = self.FindLatestModelFile(Name) 
            if Path is None:
                return False
            
        return True

############################################################################

    def MakeNeuralNetworkArchiveFullPath(self, inNeuralNetworkName : str, inEpochIndex : int) -> str:
        return self.FileNameManager.MakeFileFullPath(FileName = inNeuralNetworkName, Num = inEpochIndex)
    
    def GetLatestModelFolder(self) -> str :
        AllTimestampDirNames = self.FileNameManager.GetAllTimestampDirNames()
        if len(AllTimestampDirNames) == 0:
            return None
        
        AllTimestampDirNames.sort(key=lambda x: int(x), reverse=True)

        for DirName in range(AllTimestampDirNames):
            LatestSubFolderPath = os.path.join(self.FileNameManager.RawRootPath, DirName)
            # 获取所有子文件夹
            LeafFolders = self.FileNameManager.GetAllLeafDirNames(LatestSubFolderPath)
            if LeafFolders is None:
                continue

            for SF in LeafFolders:
                # 取最新的子文件夹
                LatestLeafFolderPath = os.path.join(LatestSubFolderPath, SF)

                # 使用 glob 以及文件名前缀来获取子文件夹下所有的 .pkl 文件
                ModelFiles = os.listdir(LatestLeafFolderPath)

                if not ModelFiles:
                    continue

                return LatestLeafFolderPath
            
        return None

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
        for Name, Model in self.NNModelDict.items():
            ModelFullPath = self.MakeNeuralNetworkArchiveFullPath(Name, inEpochIndex)
            torch.save(Model.state_dict(), ModelFullPath)
            print("Save Model:" + ModelFullPath)

    def Load(self, inForTrain : bool, inEpochIndex : int) -> None :
        pass

    def LoadLastest(self, inForTrain : bool = True):
        MaxEpochIndex = -1
        for Name, _ in self.NNModelDict.items():
            EpochIndex = self.LoadLastestByModelName(Name)
            if EpochIndex <= 0 :
                return None
            if EpochIndex > MaxEpochIndex :
                MaxEpochIndex = EpochIndex
        return MaxEpochIndex

    def LoadLastestByModelName(self, inModelName : str):
        ModelFullPath, EpochIndex = self.FindLatestModelFile(inModelName)
        if ModelFullPath is None :
            return None
        self.NNModelDict[inModelName].load_state_dict(torch.load(ModelFullPath))
        print("Load Model:" + ModelFullPath)
        return EpochIndex

    def LoadModelByTimestamp(self, inTimestamp:str, inEpochIndex):
        for Name, Model in self.NNModelDict.items():
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
