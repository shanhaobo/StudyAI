import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime

import torch.nn.functional as F

from .MultiNNTrainer import MultiNNTrainer

from KongMing.Models.BaseNNModel import BaseNNModel

class GANTrainer(MultiNNTrainer):
    def __init__(
            self,
            inGenerator : BaseNNModel,
            inDiscriminator : BaseNNModel,
            inGeneratorEmbeddingDim,
            inLearningRate = 1e-5,
            inLogRootPath="."
        ) -> None:
        super().__init__(inLearningRate, inLogRootPath)
        
        self.Generator                  = inGenerator.to(self.Device)
        self.Discriminator              = inDiscriminator.to(self.Device)

        self.GeneratorEmbeddingDim         = inGeneratorEmbeddingDim

        self.EndBatchTrain.add(self.MyEndBatchTrain)

###########################################################################################


    def _CreateOptimizer(self) -> None:
        self.Generator.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.Discriminator.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))

    def _CreateLossFN(self) -> None:
        self.Generator.ApplyLossFunc(torch.nn.BCELoss().to(self.Device))
        self.Discriminator.ApplyLossFunc(torch.nn.BCELoss().to(self.Device))

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :

        BatchSize, _, ImageHeight, ImageWidth = inBatchData.size()

        # Prepare Real and Fake Data
        RealData = inBatchData.to(self.Device)
        
        with self.Discriminator as D:
            with self.Generator as G: 
                FakeData = self.Generator(torch.randn((BatchSize, self.GeneratorEmbeddingDim, ImageHeight, ImageWidth), device=self.Device))

                DiscriminatorScores = self.Discriminator(FakeData)
                RealLabels = torch.ones(DiscriminatorScores.size(), device=self.Device)
                
                G.CalcAndAcceptLoss(DiscriminatorScores, RealLabels)

            DiscriminatorScores = self.Discriminator(RealData)
            DLossReal = D.CalcLoss(DiscriminatorScores, RealLabels)

            DiscriminatorScores = self.Discriminator(FakeData.detach())
            FakeLabels = torch.zeros(DiscriminatorScores.size(), device=self.Device)
            DLossFake = D.CalcLoss(DiscriminatorScores, FakeLabels)

            D.AcceptLoss((DLossReal + DLossFake) * 0.5)

###########################################################################################

    def MyEndBatchTrain(self, inArgs, inKVArgs) -> None:
        DLoss, _ = self.Discriminator.GetLossValue()
        GLoss, _ = self.Generator.GetLossValue()
        print(
            "{} | Epoch:{:0>4d} | Batch:{:0>4d} | DLoss:{:.8f} | GLoss:{:.8f}".
            format(
                datetime.now().strftime("[%Y/%m/%d %H:%M:%S.%f]"),
                self.CurrEpochIndex,
                self.CurrBatchIndex,
                DLoss,
                GLoss
            )
        )

###########################################################################################
