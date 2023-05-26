import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os

from .DCGANTrainer import DCGANTrainer

class GANModel(object):
    def __init__(
            self,
            inGenerator : nn.Module,
            inDiscriminator : nn.Module,
            inLatentSize,
            inLearningRate = 1e-5,
            inModelPath = "."
        ) -> None:
        self.Generator      = inGenerator
        self.Discriminator  = inDiscriminator

        self.LatentSize     = inLatentSize
        self.ModelFolderPath = inModelPath

        self.Trainer = DCGANTrainer(
            self.Generator,
            self.Discriminator,
            inLatentSize,
            inLearningRate,
            inModelPath
        )

    def Train(self, inNumEpochs : int, inDataLoader:DataLoader, inSaveModelInterval : int = 10) -> None:
        self.Trainer.Train(inNumEpochs, inDataLoader, SaveModelInterval=inSaveModelInterval)

    def Gen(self, inPostFix = ""):
        return self.Trainer.Gen(inPostFix)

    def IsExistModels(self, inForTrain : bool = True, inPostFix = "") -> bool:
        
        return self.Trainer.IsExistModels(inForTrain, inPostFix)