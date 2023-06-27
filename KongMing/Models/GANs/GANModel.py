import torch
from torch.utils.data import DataLoader

from datetime import datetime

from KongMing.Models.BaseModel import BaseModel

from KongMing.Trainer.GANTrainer import GANTrainer
from KongMing.Trainer.WGANTrainer import WGANTrainer
from KongMing.Archiver.GANArchiver import GANArchiver

class GANModel(BaseModel):
    def __init__(
            self,
            inGenerator : torch.nn.Module,
            inDiscriminator : torch.nn.Module,
            inGeneratorSize,
            inLearningRate = 1e-5,
            inModelRootFolderPath = "."
        ) -> None:
        
        NewTrainer = WGANTrainer(
            inGenerator,
            inDiscriminator,
            inGeneratorSize,
            inLearningRate
        )

        NewArchiver = GANArchiver(
            inGenerator,
            inDiscriminator,
            inModelRootFolderPath
        )

        super().__init__(NewTrainer, NewArchiver)

        self.Trainer.EndBatchTrain.add(self.EndBatchTrain)

    ###########################################################################################

    def Train(self, inDataLoader : DataLoader, inNumEpochs : int = 0, *inArgs, **inKWArgs) -> None:
        super().Train(inDataLoader, inNumEpochs, *inArgs, **inKWArgs)

    def IncTrain(self, inDataLoader : DataLoader, inNumEpochs : int = 0, *inArgs, **inKWArgs) -> None:
        super().IncTrain(inDataLoader, inNumEpochs, *inArgs, **inKWArgs)

    def Eval(self, *inArgs, **inKWArgs):
        if (super().Eval(*inArgs, **inKWArgs) == False) :
            return None
        self.Trainer.Generator.eval()
        return self.Trainer.Generator(torch.randn((1, ) + self.Trainer.GeneratorInputSize).to(self.Trainer.Device))

    ###########################################################################################

    ###########################################################################################
