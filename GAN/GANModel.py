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

        self.Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.Generator      = inGenerator.to(self.Device)
        self.Discriminator  = inDiscriminator.to(self.Device)

        self._CreateOptimizer(inOptimizerType, inLearningRate)
        
        # binary cross entropy loss and optimizer
        self.LossFN = nn.BCELoss()

        self.LatentSize = inLatentSize
        self.ModelFolderPath = inModelPath

    def Train(self, inNumEpochs : int, inDataLoader:DataLoader, inSaveModelInterval : int = 10) -> None:
        self.Generator.train()
        self.Discriminator.train()
        for i in range(inNumEpochs) :
            self._EpochTrain(inDataLoader, f"Training [EpochCount:{i}]")
            if i % inSaveModelInterval == 0 and i > 0:
                self._SaveModel(f"_{i}")
        self._SaveModel()

    def _SaveModel(self, inPostFix = "") -> None:
        if os.path.exists(self.ModelFolderPath) == False:
            os.makedirs(self.ModelFolderPath)

        torch.save(self.Generator.state_dict(), f"{self.ModelFolderPath}/Generator{inPostFix}.pkl")
        print(f"Saved:{self.ModelFolderPath}/Generator{inPostFix}.pkl")
        torch.save(self.Discriminator.state_dict(), f"{self.ModelFolderPath}/Discriminator{inPostFix}.pkl")
        print(f"Saved:{self.ModelFolderPath}/Discriminator{inPostFix}.pkl")

    def IsExistModels(self, inForTrain : bool = True, inPostFix = "") -> bool:
        bExistGModel = os.path.isfile(f"{self.ModelFolderPath}/Generator{inPostFix}.pkl")

        if inForTrain :
            bExistDModel = os.path.isfile(f"{self.ModelFolderPath}/Discriminator{inPostFix}.pkl")
        else:
            bExistDModel = True

        return bExistDModel and bExistGModel

    def _LoadModel(self, inForTrain : bool = True, inPostFix = "") -> None :
        self.Generator.load_state_dict(torch.load(f"{self.ModelFolderPath}/Generator{inPostFix}.pkl"))
        if inForTrain :
            self.Discriminator.load_state_dict(torch.load(f"{self.ModelFolderPath}/Discriminator{inPostFix}.pkl")) 

    def _CreateOptimizer(self, inOptimizerType : OptimizerType, inLearningRate) -> None:
        if inOptimizerType == OptimizerType.RMSprop :
            self.OptimizerG = optim.RMSprop(self.Generator.parameters(), lr=inLearningRate)
            self.OptimizerD = optim.RMSprop(self.Discriminator.parameters(), lr=inLearningRate)
        else : 
            self.OptimizerG = optim.Adam(self.Generator.parameters(), lr=inLearningRate, betas=(0.5, 0.999))
            self.OptimizerD = optim.Adam(self.Discriminator.parameters(), lr=inLearningRate, betas=(0.5, 0.999))

    def _CalcLossForReal(self, inBatchData):
        DiscriminatorResult = self.Discriminator(inBatchData)
        RealLabels = torch.ones(DiscriminatorResult.size()).to(self.Device)
        return self.LossFN(DiscriminatorResult, RealLabels)
    
    def _CalcLossForFake(self, inBatchData):
        DiscriminatorResult = self.Discriminator(inBatchData)
        FakeLabels = torch.zeros(DiscriminatorResult.size()).to(self.Device)
        return self.LossFN(DiscriminatorResult, FakeLabels)
    
    def _BackPropagate(self, inOptimizer, inLoss):
        inOptimizer.zero_grad()
        inLoss.backward()
        inOptimizer.step()
    
    def _EpochTrain(self, inDataLoader:DataLoader, inCurrEpochInfo) -> None:
        for i, (RealBatchData, _) in enumerate(inDataLoader):
            nBatchSize = RealBatchData.size(0)
            BatchLatentSize = (nBatchSize, ) + self.LatentSize
            
            # Optimize Discriminator
            RealBatchData = RealBatchData.to(self.Device)
            DLossReal = self._CalcLossForReal(RealBatchData)

            FakeBatchData = self.Generator(torch.randn(BatchLatentSize).to(self.Device))
            DLossFake = self._CalcLossForFake(FakeBatchData)

            DLoss = (DLossReal + DLossFake) / 2
            self._BackPropagate(self.OptimizerD, DLoss)
            
            # Optimize Generator
            FakeBatchData = self.Generator(torch.randn(BatchLatentSize).to(self.Device))
            GLoss = self._CalcLossForReal(FakeBatchData)
            self._BackPropagate(self.OptimizerG, GLoss)

            print(f"[{inCurrEpochInfo}][BatchCount:{i}] [Discriminator Loss:{DLoss.item()}] [Generator Loss:{GLoss.item()}]")

    def Gen(self, inPostFix = ""):
        self._LoadModel(inForTrain=False, inPostFix=inPostFix)
        self.Generator.eval()
        return self.Generator(torch.randn((1, ) + self.LatentSize).to(self.Device))
