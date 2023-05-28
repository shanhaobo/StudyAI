import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from functools import partial

import os

from Trainer.GANTrainer import GANTrainer
from Utils.Archiver import GANArchiver

class GANModel(object):
    def __init__(
            self,
            inGenerator : nn.Module,
            inDiscriminator : nn.Module,
            inGeneratorSize,
            inLearningRate = 1e-5,
            inModelFolderPath = "."
        ) -> None:
        
        self.Trainer = GANTrainer(
            inGenerator,
            inDiscriminator,
            inGeneratorSize,
            inLearningRate
        )

        self.Archiver = GANArchiver(
            inGenerator,
            inDiscriminator,
            inModelFolderPath
        )

        self.Trainer.EndBatchTrain.add(self.EndBatchTrain)
        self.Trainer.EndEpochTrain.add(self.EndEpochTrain)
        self.Trainer.EndTrain.add(self.EndTrain)

        self.Generator = self.Trainer.Generator

    def Train(self, inNumEpochs : int, inDataLoader:DataLoader, inSaveModelInterval : int = 10) -> None:
        self.Trainer.Train(inNumEpochs, inDataLoader, SaveModelInterval=inSaveModelInterval)

    def Gen(self, inPostFix = ""):
        self.Archiver.Load(inForTrain=False, inPostfix=inPostFix)
        self.Generator.eval()
        return self.Generator(torch.randn((1, ) + self.Trainer.GeneratorInputSize).to(self.Trainer.Device))

    def IsExistModels(self, inForTrain : bool = True, inPostFix = "") -> bool:
        return self.Archiver.IsExistModel(inForTrain, inPostFix)
    
    def EndBatchTrain(self, *inArgs, **inKWArgs) -> None:
        print(
            "Epoch:{:0 > 4d} | Batch:{:0 > 6d} | DLoss:{:.8f} | GLoss:{:.8f}".
            format(self.Trainer.CurrEpochIndex + 1, self.Trainer.CurrBatchIndex + 1, self.Trainer.CurrBatchDiscriminatorLoss, self.Trainer.CurrBatchGeneratorLoss)
        )
        pass

    def EndEpochTrain(self, *inArgs, **inKWArgs) -> None:
        if self.Trainer.CurrEpochIndex % 2 and self.Trainer.CurrEpochIndex > 0 :
            print("Epoch:{} Save Models".format(self.Trainer.CurrEpochIndex + 1))
            self.Archiver.Save("{:0 > 4d}".format(self.Trainer.CurrEpochIndex + 1))
        pass

    def EndTrain(self, *inArgs, **inKWArgs)->None:
        self.Archiver.Save(f"{self.Trainer.CurrEpochIndex + 1}")
