import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os

from enum import Enum

class OptimizerType(Enum):
    Adam = 1
    RMSprop = 2

class GANModel(object):
    def __init__(
            self,
            inGenerator : nn.Module,
            inDiscriminator : nn.Module,
            inLatentSize,
            inOptimizerType : OptimizerType = OptimizerType.Adam,
            inLearningRate = 1e-5,
            inModelPath = "."
        ) -> None:
        self.Generator      = inGenerator
        self.Discriminator  = inDiscriminator

        self._CreateOptimizer(inOptimizerType, inLearningRate)
        
        # binary cross entropy loss and optimizer
        self.LossFN = nn.BCELoss()

        self.LatentSize = inLatentSize
        self.ModelPath = inModelPath

    def Train(self, inNumEpochs : int, inDataLoader:DataLoader) -> None:
        self.Generator.train()
        self.Discriminator.train()
        for i in range(inNumEpochs) :
            self._BatchTrain(inDataLoader)
            if i % 100 == 0 and i > 0:
                self._SaveModel(f"_{i}")
        self._SaveModel()

    def _SaveModel(self, inPostFix = "") -> None:
        torch.save(self.Generator.state_dict(), f"{self.ModelPath}/Generator{inPostFix}.pkl")
        print(f"Saved:{self.ModelPath}/Generator{inPostFix}.pkl")
        torch.save(self.Discriminator.state_dict(), f"{self.ModelPath}/Discriminator{inPostFix}.pkl")
        print(f"Saved:{self.ModelPath}/Discriminator{inPostFix}.pkl")

    def IsExistModels(self, inForTrain : bool = True, inPostFix = "") -> bool:
        bExistGModel = os.path.isfile(f"{self.ModelPath}/Generator{inPostFix}.pkl")

        if inForTrain :
            bExistDModel = os.path.isfile(f"{self.ModelPath}/Discriminator{inPostFix}.pkl")
        else:
            bExistDModel = True

        return bExistDModel and bExistGModel

    def _LoadModel(self, inForTrain : bool = True, inPostFix = "") -> None :
        self.Generator.load_state_dict(torch.load(f"{self.ModelPath}/Generator{inPostFix}.pkl"))
        if inForTrain :
            self.Discriminator.load_state_dict(torch.load(f"{self.ModelPath}/Discriminator{inPostFix}.pkl")) 

    def _CreateOptimizer(self, inOptimizerType : OptimizerType, inLearningRate) -> None:
        if inOptimizerType == OptimizerType.RMSprop :
            self.OptimizerG = optim.RMSprop(self.Generator.parameters(), lr=inLearningRate)
            self.OptimizerD = optim.RMSprop(self.Discriminator.parameters(), lr=inLearningRate)
        else : 
            self.OptimizerG = optim.Adam(self.Generator.parameters(), lr=inLearningRate, betas=(0.5, 0.999))
            self.OptimizerD = optim.Adam(self.Discriminator.parameters(), lr=inLearningRate, betas=(0.5, 0.999))

    def _BatchTrain(self, inDataLoader:DataLoader) -> None:
        nBatchSize = inDataLoader.batch_size
        BatchLatentSize = (nBatchSize, ) + self.LatentSize
        for i, BatchData in enumerate(inDataLoader):
            # Optimize Discriminator
            Outputs = self.Discriminator(BatchData)
            RealLabels = torch.ones(Outputs.size())
            DLossReal = self.LossFN(Outputs, RealLabels)

            FakeBatchData = self.Generator(torch.randn(BatchLatentSize))
            Outputs = self.Discriminator(FakeBatchData)
            FakeLabels = torch.zeros(Outputs.size())
            DLossFake = self.LossFN(Outputs, FakeLabels)

            DLoss = (DLossReal + DLossFake) / 2

            self.OptimizerD.zero_grad()
            DLoss.backward()
            self.OptimizerD.step()

            # Optimize Generator
            Outputs = self.Discriminator(FakeBatchData.detach())
            RealLabels = torch.ones(Outputs.size())
            GLoss = self.LossFN(Outputs, RealLabels)

            self.OptimizerG.zero_grad()
            GLoss.backward()
            self.OptimizerG.step()

    def Gen(self, inPostFix = "") -> None:
        self._LoadModel(inForTrain=True, inPostFix=inPostFix)
        self.Generator.eval()
        return self.Generator(torch.randn((1, ) + self.LatentSize))
