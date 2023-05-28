import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Trainer.GANTrainer import GANTrainer

import os

class DCGANTrainer(GANTrainer):
    def __init__(
            self,
            inGenerator : nn.Module,
            inDiscriminator : nn.Module,
            inGeneratorInputSize,
            inLearningRate = 1e-5,
            inModelPath = "."
        ) -> None:
        super().__init__(inGenerator, inDiscriminator, inGeneratorInputSize, inLearningRate)

        self.ModelFolderPath = inModelPath

    def _SaveModel(self, inPostFix = "") -> None:
        if os.path.exists(self.ModelFolderPath) == False:
            os.makedirs(self.ModelFolderPath)

        torch.save(self.Generator.state_dict(), f"{self.ModelFolderPath}/Generator{inPostFix}.pkl")
        print(f"Saved:{self.ModelFolderPath}/Generator{inPostFix}.pkl")
        torch.save(self.Discriminator.state_dict(), f"{self.ModelFolderPath}/Discriminator{inPostFix}.pkl")
        print(f"Saved:{self.ModelFolderPath}/Discriminator{inPostFix}.pkl")

    def _LoadModel(self, inForTrain : bool = True, inPostFix = "") -> None :
        self.Generator.load_state_dict(torch.load(f"{self.ModelFolderPath}/Generator{inPostFix}.pkl"))
        if inForTrain :
            self.Discriminator.load_state_dict(torch.load(f"{self.ModelFolderPath}/Discriminator{inPostFix}.pkl")) 

    def Gen(self, inPostFix = ""):
        self._LoadModel(inForTrain=False, inPostFix=inPostFix)
        self.Generator.eval()
        return self.Generator(torch.randn((1, ) + self.GeneratorInputSize).to(self.Device))

    def IsExistModels(self, inForTrain : bool = True, inPostFix = "") -> bool:
        bExistGModel = os.path.isfile(f"{self.ModelFolderPath}/Generator{inPostFix}.pkl")

        if inForTrain :
            bExistDModel = os.path.isfile(f"{self.ModelFolderPath}/Discriminator{inPostFix}.pkl")
        else:
            bExistDModel = True

        return bExistDModel and bExistGModel

    def _EndBatchTrain(self, inBatchIndex, **inArgs) -> None:
        print(inArgs)
        pass

    def _EndEpochTrain(self, inEpochIndex, **inArgs) -> None:
        print(inArgs)
        if inEpochIndex % 2 == 0 and inEpochIndex > 0 :
            self._SaveModel(f"_{inEpochIndex}")
        pass

    def _EndTrain(self, **inArgs) -> None:
        self._SaveModel()
        pass
