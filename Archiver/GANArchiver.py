import torch

from .MultiNNArchiver import MultiNNArchiver

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
