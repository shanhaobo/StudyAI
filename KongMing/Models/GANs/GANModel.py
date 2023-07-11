import torch
from torch.utils.data import DataLoader

from datetime import datetime

from KongMing.Models.BaseModel import BaseModel

from KongMing.Trainer.GANTrainer import GANTrainer
from KongMing.Trainer.WGANTrainer import WGANTrainer
from KongMing.Archiver.GANArchiver import GANArchiver

from KongMing.Utils.CaseInsensitiveContainer import CaseInsensitiveList, CaseInsensitiveDict

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

        g = self._SumParameters(inGenerator)
        d = self._SumParameters(inDiscriminator)
        print("Sum of Params:{:,} | Generator Params:{:,} | Discriminator Params:{:,}".format(g + d, g, d))

    ###########################################################################################

    def Eval(self, inEpoch, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None):
        if (super().Eval(inEpoch, inArgs, inKVArgs) == False) :
            return None
        
        BatchSize = inKVArgs.get("inBatchSize")
        if (BatchSize is None) :
            BatchSize = 1

        self.Trainer.Generator.eval()
        return self.Trainer.Generator(torch.randn((BatchSize, ) + self.Trainer.GeneratorInputSize).to(self.Trainer.Device))

    ###########################################################################################
