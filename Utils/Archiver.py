import torch
import os

class BaseArchiver(object):
    def __init__(self, inModelFolderPath : str = ".") -> None:
        self.ModelFolderPath = inModelFolderPath

    def Save(self, inPostFix = "") -> None:
        pass
    def Load(self, inForTrain : bool = True, inPostFix = "") -> None :
        pass
    def IsExistModel(self, inForTrain : bool = True, inPostFix = "") -> bool:
        pass

class MultiNNArchiver(BaseArchiver):
    def __init__(self, inModelFolderPath : str = ".") -> None:
        super().__init__(inModelFolderPath)

class GANArchiver(MultiNNArchiver):
    def __init__(self, inGenerator : torch.nn.Module, inDiscriminator : torch.nn.Module, inModelFolderPath : str = ".") -> None:
        super().__init__(inModelFolderPath)
        self.Generator = inGenerator
        self.Discriminator = inDiscriminator

    def Save(self, inPostfix = "") -> None:
        if os.path.exists(self.ModelFolderPath) == False:
            os.makedirs(self.ModelFolderPath)

        torch.save(self.Generator.state_dict(), f"{self.ModelFolderPath}/Generator{inPostfix}.pkl")
        print(f"Saved:{self.ModelFolderPath}/Generator{inPostfix}.pkl")
        torch.save(self.Discriminator.state_dict(), f"{self.ModelFolderPath}/Discriminator{inPostfix}.pkl")
        print(f"Saved:{self.ModelFolderPath}/Discriminator{inPostfix}.pkl")


    def IsExistModel(self, inForTrain : bool = True, inPostfix = "") -> bool:
        bExistGModel = os.path.isfile(f"{self.ModelFolderPath}/Generator{inPostfix}.pkl")

        if inForTrain :
            bExistDModel = os.path.isfile(f"{self.ModelFolderPath}/Discriminator{inPostfix}.pkl")
        else:
            bExistDModel = True

        return bExistDModel and bExistGModel

    def Load(self, inForTrain : bool = True, inPostfix = "") -> None :
        self.Generator.load_state_dict(torch.load(f"{self.ModelFolderPath}/Generator{inPostfix}.pkl"))
        if inForTrain :
            self.Discriminator.load_state_dict(torch.load(f"{self.ModelFolderPath}/Discriminator{inPostfix}.pkl")) 
