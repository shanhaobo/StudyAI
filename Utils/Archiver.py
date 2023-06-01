import torch
import os
import glob
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
    
    def IsExistModel(self, inForTrain : bool = True, *inArgs, **inKWArgs) -> bool:
        pass

    def GetModelFullPath(self, inModelName : str, inEpochIndex : int,  inSuffix : str = "") -> str:
        return os.path.join(self.ModelArchiveFolderPath, "{}_{}_{}.pkl".format(inModelName, inEpochIndex, inSuffix))
    
    def FindLatestModelFile(self, inModelName : str):
        # 获取所有子文件夹
        SubFolders = [d for d in os.listdir(self.ModelArchiveRootFolderPath) if os.path.isdir(os.path.join(self.ModelArchiveRootFolderPath, d))]

        # 按照时间戳排序子文件夹（从最新到最旧）
        SubFolders.sort(reverse=True)

        if not SubFolders:
            return None, None

        for SF in SubFolders:
            # 取最新的子文件夹
            LatestSubFolder = SF
            LatestSubFolderPath = os.path.join(self.ModelArchiveRootFolderPath, LatestSubFolder)

            # 使用 glob 以及文件名前缀来获取子文件夹下所有的 .pkl 文件
            ModelFiles = glob.glob(os.path.join(LatestSubFolderPath, f"{inModelName}*.pkl"))

            if not ModelFiles:
                continue

            # 返回数字最大（也就是最新）的文件
            return FindFileWithMaxNum(ModelFiles, inModelName, "*", "pkl")
        
        return None, None

class MultiNNArchiver(BaseArchiver):
    def __init__(self, inModelPrefix : str, inModelRootFolderPath : str = ".", inNeedTimestamp : bool = True) -> None:
        super().__init__(inModelPrefix, inModelRootFolderPath, inNeedTimestamp)
        self.NNModelDict = {}

    def Save(self, inEpochIndex : int, inSuffix = "") -> None:
        for Name, Model in self.NNModelDict.items():
            ModelFullPath = self.GetModelFullPath(Name, inEpochIndex, inSuffix)
            torch.save(Model.state_dict(), ModelFullPath)
            print("Save Model:" + ModelFullPath)

    def LoadLastest(self, inForTrain : bool = True):
        MaxEpochIndex = -1
        for Name, _ in self.NNModelDict.items():
            bSuccess, EpochIndex = self.LoadLastestByModelName(Name)
            if bSuccess == False :
                return False, -1
            if EpochIndex > MaxEpochIndex :
                MaxEpochIndex = EpochIndex
        return True, MaxEpochIndex

    def LoadLastestByModelName(self, inModelName : str):
        ModelFullPath, EpochIndex = self.FindLatestModelFile(inModelName)
        if ModelFullPath == None :
            return False, -1
        self.NNModelDict[inModelName].load_state_dict(torch.load(ModelFullPath))
        print("Load Model:" + ModelFullPath)
        return True, EpochIndex

class GANArchiver(MultiNNArchiver):
    def __init__(
            self,
            inGenerator : torch.nn.Module,
            inDiscriminator : torch.nn.Module,
            inModelPrefix : str = "GAN",
            inModelRootFolderPath : str = ".",
            inTimestamp : bool = True
        ) -> None:
        super().__init__(inModelPrefix, inModelRootFolderPath, inTimestamp)
        self.Generator = inGenerator
        self.Discriminator = inDiscriminator

        self.NNModelDict["Generator"] = self.Generator
        self.NNModelDict["Discriminator"] = self.Discriminator

    def IsExistModel(self, inForTrain : bool = True, *inArgs, **inKWArgs) -> bool:
        if ((self.FindLatestModelFile("Generator") != None) and (inForTrain == False)) :
            return True
        
        return self.FindLatestModelFile("Discriminator") != None

    def Load(self, inForTrain : bool = True, inSuffix = "") -> None :
        self.Generator.load_state_dict(torch.load(f"{self.ModelRootFolderPath}/Generator{inSuffix}.pkl"))
        if inForTrain :
            self.Discriminator.load_state_dict(torch.load(f"{self.ModelRootFolderPath}/Discriminator{inSuffix}.pkl")) 

    def LoadLastest(self, inForTrain : bool = False) -> bool:
        bSuccess, EpochIndex = self.LoadLastestByModelName("Generator")
        if bSuccess == False :
            return False, -1
        
        if inForTrain :
            bSuccess, EpochIndex =  self.LoadLastestByModelName("Discriminator")
            return False, -1

        return True, EpochIndex
