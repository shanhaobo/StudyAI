import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime

from .MultiNNTrainer import MultiNNTrainer

from KongMing.Modules.BaseNNModule import BaseNNModule

class GANTrainer(MultiNNTrainer):
    def __init__(
            self,
            inGenerator : BaseNNModule,
            inDiscriminator : BaseNNModule,
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
        #self.OptimizerG = optim.Adam(self.Generator.parameters(),     lr=self.LearningRate, betas=(0.5, 0.999))
        #self.OptimizerD = optim.Adam(self.Discriminator.parameters(), lr=self.LearningRate, betas=(0.5, 0.999))
        self.Generator.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.Discriminator.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))

    def _CreateLossFN(self) -> None:
        #self.LossFN     = nn.BCELoss().to(self.Device)
        self.Generator.ApplyLossFunc(nn.BCELoss().to(self.Device))
        self.Discriminator.ApplyLossFunc(nn.BCELoss().to(self.Device))

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :

        BatchSize, _, ImageHeight, ImageWidth = inBatchData.size()
        # Prepare Real and Fake Data
        RealData = inBatchData.to(self.Device)

        with self.Generator as G, self.Discriminator as D:
            
            FakeData = self.Generator(torch.randn((BatchSize, self.GeneratorEmbeddingDim, ImageHeight, ImageWidth), device=self.Device))
            
            DiscriminatorScores = self.Discriminator(RealData)
            RealLabels = torch.ones(DiscriminatorScores.size(), device=self.Device)
            DLossReal = D.CalcLoss(DiscriminatorScores, RealLabels)

            DiscriminatorScores = self.Discriminator(FakeData.detach())
            FakeLabels = torch.zeros(DiscriminatorScores.size(), device=self.Device)
            DLossFake = D.CalcLoss(DiscriminatorScores, FakeLabels)

            D.ApplyLoss((DLossReal + DLossFake) * 0.5)

            DiscriminatorScores = self.Discriminator(FakeData)
            RealLabels = torch.ones(DiscriminatorScores.size(), device=self.Device)
            G.ApplyCalcLoss(DiscriminatorScores, RealLabels)


###########################################################################################

    def MyEndBatchTrain(self, inArgs, inKVArgs) -> None:
        DLoss, _ = self.Discriminator.GetLoss()
        GLoss, _ = self.Generator.GetLoss()
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
