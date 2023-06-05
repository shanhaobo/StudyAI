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

        self.FileNameManager = FileManagerWithNum(self.ModelArchiveRootFolderPath, ".pkl", 100)

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
        # 获取所有子文件夹
        SubFolders = self.FileNameManager.GetAllLeafDirNames()

        if not SubFolders:
            return None

        for SF in SubFolders:
            # 取最新的子文件夹
            LatestSubFolderPath = os.path.join(self.ModelArchiveRootFolderPath, SF)

            # 使用 glob 以及文件名前缀来获取子文件夹下所有的 .pkl 文件
            ModelFiles = os.listdir(LatestSubFolderPath)

            if not ModelFiles:
                continue

            return LatestSubFolderPath
        
        return None

    def FindLatestModelFile(self, inModelName : str):
        LatestFolderPath = self.GetLatestModelFolder()

         # 返回数字最大（也就是最新）的文件
        FileName, MaxNum =  FindFileWithMaxNum(os.listdir(LatestFolderPath), inModelName, "*", "pkl")
        if not FileName:
            return None, None
        return os.path.join(LatestFolderPath, FileName), MaxNum
