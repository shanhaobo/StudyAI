import torch
import os
import glob
from datetime import datetime


class BaseArchiver(object):
    def __init__(self, inModelPrefix : str, inModelRootFolderPath : str = ".", inNeedTimestamp : bool = True) -> None:
        self.ModelPrefix                = inModelPrefix
        self.ModelRootFolderPath        = inModelRootFolderPath
        self.ModelArchiveRootFolderPath = os.path.join(self.ModelRootFolderPath, self.ModelPrefix)
        self.ModelArchiveFolderPath     = self.ModelArchiveRootFolderPath

        if inNeedTimestamp :
            NowStr = datetime.now().strftime("%Y%m%d%H%M")
            self.ModelArchiveFolderPath = os.path.join(self.ModelArchiveFolderPath, NowStr)

        os.makedirs(self.ModelArchiveFolderPath)

    def Save(self, inSuffix = "") -> None:
        pass
    
    def Load(self, inForTrain : bool = True, inSuffix = "") -> None :
        pass

    def LoadLastest(self, inForTrain : bool = True) -> bool:
        pass
    
    def IsExistModel(self, inForTrain : bool = True, inSuffix = "") -> bool:
        pass

    def GetModelFullPath(self, inModelName : str, inSuffix : str = "") -> str:
        return os.path.join(self.ModelArchiveFolderPath, "{}_{}.pkl".format(inModelName, inSuffix))
    
    def FindLatestModelFile(self, inModelName : str):
        # 获取所有子文件夹
        SubFolders = [d for d in os.listdir(self.ModelArchiveRootFolderPath) if os.path.isdir(os.path.join(self.ModelArchiveFolderPath, d))]

        # 按照时间戳排序子文件夹（从最新到最旧）
        SubFolders.sort(reverse=True)

        if not SubFolders:
            return None

        # 取最新的子文件夹
        LatestSubFolder = SubFolders[0]
        LatestSubFolderPath = os.path.join(self.ModelArchiveRootFolderPath, LatestSubFolder)

        # 使用 glob 以及文件名前缀来获取子文件夹下所有的 .pkl 文件
        ModelFiles = glob.glob(os.path.join(LatestSubFolderPath, f'{inModelName}*.pkl'))

        if not ModelFiles:
            return None

        # 从文件名中获取数字部分，并转换为int，然后排序
        ModelFiles.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True)

        # 返回数字最大（也就是最新）的文件
        return ModelFiles[0]

    def FindLatestModelFiles(self, inModelNames):
        # 获取所有子文件夹
        SubFolders = [d for d in os.listdir(self.ModelArchiveRootFolderPath) if os.path.isdir(os.path.join(self.ModelArchiveRootFolderPath, d))]

        # 按照时间戳排序子文件夹（从最新到最旧）
        SubFolders.sort(reverse=True)

        if not SubFolders:
            return None

        # 取最新的子文件夹
        LatestSubFolder = SubFolders[0]
        LatestSubFolderPath = os.path.join(self.ModelArchiveRootFolderPath, LatestSubFolder)

        LatestModelFiles = {}

        for ModelName in inModelNames:
            # 使用 glob 以及文件名前缀来获取子文件夹下所有的 .pkl 文件
            ModelFiles = glob.glob(os.path.join(LatestSubFolderPath, f'{ModelName}*.pkl'))

            if not ModelFiles:
                LatestModelFiles[ModelName] = None
            else:
                # 从文件名中获取数字部分，并转换为int，然后排序
                ModelFiles.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True)

                # 记录数字最大（也就是最新）的文件
                LatestModelFiles[ModelName] = ModelFiles[0]

        return LatestModelFiles

class MultiNNArchiver(BaseArchiver):
    def __init__(self, inModelPrefix : str, inModelRootFolderPath : str = ".", inNeedTimestamp : bool = True) -> None:
        super().__init__(inModelPrefix, inModelRootFolderPath, inNeedTimestamp)
        self.NNModelDict = {}

    def Save(self, inSuffix = "") -> None:
        for Name, Model in self.NNModelDict.items():
            ModelFullPath = self.GetModelFullPath(Name, inSuffix)
            torch.save(Model.state_dict(), ModelFullPath)
            print("Save Model:" + ModelFullPath)

    def LoadLastest(self, inForTrain : bool = True) -> bool:
        for Name, _ in self.NNModelDict.items():
            if self.LoadLastestByModelName(Name) == False:
                return False
        return True

    def LoadLastestByModelName(self, inModelName : str) -> bool:
        ModelFullPath = self.FindLatestModelFile(inModelName)
        if ModelFullPath == None :
            return False
        self.NNModelDict[inModelName].load_state_dict(torch.load(ModelFullPath))
        print("Load Model:" + ModelFullPath)
        return True

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

    def IsExistModel(self, inForTrain : bool = True, inSuffix = "") -> bool:
        bExistGModel = os.path.isfile(f"{self.ModelRootFolderPath}/Generator{inSuffix}.pkl")
        if inForTrain :
            bExistDModel = os.path.isfile(f"{self.ModelRootFolderPath}/Discriminator{inSuffix}.pkl")
        else:
            bExistDModel = True
        return bExistDModel and bExistGModel

    def Load(self, inForTrain : bool = True, inSuffix = "") -> None :
        self.Generator.load_state_dict(torch.load(f"{self.ModelRootFolderPath}/Generator{inSuffix}.pkl"))
        if inForTrain :
            self.Discriminator.load_state_dict(torch.load(f"{self.ModelRootFolderPath}/Discriminator{inSuffix}.pkl")) 

    def LoadLastest(self, inForTrain : bool = False) -> bool:
        if self.LoadLastestByModelName("Generator") == False :
            return False
        
        if inForTrain and self.LoadLastestByModelName("Discriminator") == False :
            return False

        return True
