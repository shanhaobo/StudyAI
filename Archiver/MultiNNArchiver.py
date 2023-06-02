import torch

from .BaseArchiver import BaseArchiver

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
            if bSuccess == False :
                return False, -1

        return True, EpochIndex
