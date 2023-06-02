import os
from datetime import datetime
from Utils.ModelFileOp import FindFileWithMaxNum

class BaseArchiver(object):
    def __init__(self, inModelPrefix : str, inModelRootFolderPath : str = ".", inNeedTimestamp : bool = True) -> None:
        self.ModelPrefix                = inModelPrefix
        self.ModelRootFolderPath        = inModelRootFolderPath
        self.ModelArchiveRootFolderPath = os.path.join(self.ModelRootFolderPath, self.ModelPrefix)
        self.ModelArchiveFolderPath     = self.ModelArchiveRootFolderPath

        if inNeedTimestamp :
            NowStr = datetime.now().strftime("%Y%m%d%H%M")
            self.ModelArchiveFolderPath = os.path.join(self.ModelArchiveFolderPath, NowStr)

        os.makedirs(self.ModelArchiveFolderPath, exist_ok=True)

    def Save(self, inEpochIndex : int, inSuffix = "") -> None:
        pass
    
    def Load(self, inForTrain : bool = True, inSuffix = "") -> None :
        pass

    def LoadLastest(self, inForTrain : bool = True) -> bool:
        pass

    def LoadLastestByModelName(self, inModelName : str):
        pass
    
    def IsExistModel(self, inForTrain : bool = True, *inArgs, **inKWArgs) -> bool:
        pass

    def GetModelFullPath(self, inModelName : str, inEpochIndex : int,  inSuffix : str = "", inExtension = "pkl") -> str:
        return os.path.join(
            self.ModelArchiveFolderPath,
            "{}_{:0>6d}_{}.{}}".format(inModelName, inEpochIndex, inSuffix, inExtension)
        )
    
    def GetLatestModelFolder(self) :
        # 获取所有子文件夹
        SubFolders = [f for f in os.listdir(self.ModelArchiveRootFolderPath) if os.path.isdir(os.path.join(self.ModelArchiveRootFolderPath, f))]

        # 按照时间戳排序子文件夹（从最新到最旧）
        SubFolders.sort(reverse=True)

        if not SubFolders:
            return None

        for SF in SubFolders:
            # 取最新的子文件夹
            LatestSubFolder = SF
            LatestSubFolderPath = os.path.join(self.ModelArchiveRootFolderPath, LatestSubFolder)

            # 使用 glob 以及文件名前缀来获取子文件夹下所有的 .pkl 文件
            ModelFiles = os.listdir(LatestSubFolderPath)

            if not ModelFiles:
                continue

            return ModelFiles
        
        return None



    def FindLatestModelFile(self, inModelName : str):

        # 获取所有子文件夹
        SubFolders = [f for f in os.listdir(self.ModelArchiveRootFolderPath) if os.path.isdir(os.path.join(self.ModelArchiveRootFolderPath, f))]

        # 按照时间戳排序子文件夹（从最新到最旧）
        SubFolders.sort(reverse=True)

        if not SubFolders:
            return None, None

        for SF in SubFolders:
            # 取最新的子文件夹
            LatestSubFolder = SF
            LatestSubFolderPath = os.path.join(self.ModelArchiveRootFolderPath, LatestSubFolder)

            # 使用 glob 以及文件名前缀来获取子文件夹下所有的 .pkl 文件
            ModelFiles = os.listdir(LatestSubFolderPath)

            if not ModelFiles:
                continue

            # 返回数字最大（也就是最新）的文件
            FlieName, MaxNum =  FindFileWithMaxNum(ModelFiles, inModelName, "*", "pkl")
            return os.path.join(LatestSubFolderPath, FlieName), MaxNum
        
        return None, None
