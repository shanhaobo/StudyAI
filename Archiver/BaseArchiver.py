import os
from datetime import datetime
from Utils.ModelFileOp import FindFileWithMaxNum

from .Path.FileManagerWithNum import FileManagerWithNum

class BaseArchiver(object):
    def __init__(self, inModelPrefix : str, inModelRootFolderPath : str = ".") -> None:
        self.ModelPrefix                = inModelPrefix
        self.ModelRootFolderPath        = inModelRootFolderPath
        self.ModelArchiveRootFolderPath = os.path.join(self.ModelRootFolderPath, self.ModelPrefix)
        self.ModelArchiveFolderPath     = self.ModelArchiveRootFolderPath

        self.FileNameManager = FileManagerWithNum(self.ModelArchiveRootFolderPath, ".pkl", 100, True)

    def Save(self, inEpochIndex : int, inSuffix : str) -> None:
        pass
    
    def Load(self, inForTrain : bool, inEpochIndex : int, inSuffix : str) -> None :
        pass

    def LoadLastest(self, inForTrain : bool, inSuffix : str) -> int:
        self.Load(inForTrain, -1, inSuffix)
        pass

    def LoadLastestByModelName(self, inModelName : str):
        pass
    
    def IsExistModel(self, inForTrain : bool = True, *inArgs, **inKWArgs) -> bool:
        pass

    def MakeNeuralNetworkArchiveFullPath(self, inNeuralNetworkName : str, inEpochIndex : int) -> str:
        return self.FileNameManager.MakeFileFullPath(FileName = inNeuralNetworkName, Num = inEpochIndex)
    
    def GetLatestModelFolder(self) -> str :
        LatestSubFolderPath = self.FileNameManager.GetLatestTimestampDirPath()
        if not LatestSubFolderPath:
            return None

        # 获取所有子文件夹
        LeafFolders = self.FileNameManager.GetAllLeafDirNames(LatestSubFolderPath)
        if not LeafFolders:
            return None

        LeafFolders.sort(key=lambda x: int(x), reverse=True)

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

         # 返回数字最大（也就是最新）的文件
        FileName, MaxNum =  FindFileWithMaxNum(os.listdir(LatestFolderPath), inModelName, "*", "pkl")
        if not FileName:
            return None, None
        return os.path.join(LatestFolderPath, FileName), MaxNum
