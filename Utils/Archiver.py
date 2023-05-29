import torch
import os
from datetime import datetime


class BaseArchiver(object):
    def __init__(self, inModelPrefix : str, inModelRootFolderPath : str = ".", inNeedTimestamp : bool = True) -> None:
        self.ModelPrefix                = inModelPrefix
        self.ModelRootFolderPath        = inModelRootFolderPath
        self.ModelArchiverFolderPath    = os.path.join(self.ModelRootFolderPath, self.ModelPrefix)

        if inNeedTimestamp :
            now = datetime.now()
            self.ModelArchiverFolderPath = os.path.join(self.ModelArchiverFolderPath, now.strftime("%Y%m%d%H%M"))

        os.makedirs(self.ModelArchiverFolderPath)

    def Save(self, inSuffix = "") -> None:
        pass
    
    def Load(self, inForTrain : bool = True, inSuffix = "") -> None :
        pass
    
    def IsExistModel(self, inForTrain : bool = True, inSuffix = "") -> bool:
        pass

    def GetModelFullPath(self, inModelName : str, inSuffix : str = "") -> str:
        return os.path.join(self.ModelArchiverFolderPath, "{}_{}.pkl".format(inModelName, inSuffix))

class MultiNNArchiver(BaseArchiver):
    def __init__(self, inModelPrefix : str, inModelRootFolderPath : str = ".", inNeedTimestamp : bool = True) -> None:
        super().__init__(inModelPrefix, inModelRootFolderPath, inNeedTimestamp)
        self.NNModelDict = {}

    def Save(self, inSuffix = "") -> None:
        for Name, Model in self.NNModelDict.items():
            ModelFullPath = self.GetModelFullPath(Name, inSuffix)
            torch.save(Model.state_dict(), ModelFullPath)
            print(ModelFullPath)


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
