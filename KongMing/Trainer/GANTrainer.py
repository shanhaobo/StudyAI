import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime

from .MultiNNTrainer import MultiNNTrainer

class GANTrainer(MultiNNTrainer):
    def __init__(
            self,
            inGenerator : nn.Module,
            inDiscriminator : nn.Module,
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
        self.OptimizerG = optim.Adam(self.Generator.parameters(),     lr=self.LearningRate, betas=(0.5, 0.999))
        self.OptimizerD = optim.Adam(self.Discriminator.parameters(), lr=self.LearningRate, betas=(0.5, 0.999))
        pass

    def _CreateLossFN(self) -> None:
        self.LossFN     = nn.BCELoss().to(self.Device)
        pass

    def _CalcLossForReal(self, inBatchData):
        DiscriminatorScores = self.Discriminator(inBatchData)
        RealLabels = torch.ones(DiscriminatorScores.size(), device=self.Device)
        return self.LossFN(DiscriminatorScores, RealLabels)
    
    def _CalcLossForFake(self, inBatchData):
        DiscriminatorScores = self.Discriminator(inBatchData)
        FakeLabels = torch.zeros(DiscriminatorScores.size(), device=self.Device)
        return self.LossFN(DiscriminatorScores, FakeLabels)


    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :
        BatchSize, _, ImageHeight, ImageWidth = inBatchData.size()
        # Prepare Real and Fake Data
        RealData = inBatchData.to(self.Device)

        self._BeginBackPropagate(self.OptimizerD)
        self._BeginBackPropagate(self.OptimizerG)
        
        FakeData = self.Generator(torch.randn((BatchSize, self.GeneratorEmbeddingDim, ImageHeight, ImageWidth), device=self.Device))

        # Calc Score or Loss
        DLossReal = self._CalcLossForReal(RealData)
        # detach() do not effect Generator
        DLossFake = self._CalcLossForFake(FakeData.detach())

        DLoss = (DLossReal + DLossFake) * 0.5

        # Optimize Discriminator
        self._EndBackPropagate(self.OptimizerD, DLoss)
        
        # Optimize Generator
        GLoss = self._CalcLossForReal(FakeData) 
        self._EndBackPropagate(self.OptimizerG, GLoss)

        self.CurrBatchDiscriminatorLoss = DLoss.item()
        self.CurrBatchGeneratorLoss     = GLoss.item()

        pass

###########################################################################################

    def MyEndBatchTrain(self, inArgs, inKVArgs) -> None:
        NowStr  = datetime.now().strftime("[%Y/%m/%d %H:%M:%S.%f]")
        print(
            "{} | Epoch:{:0>4d} | Batch:{:0>4d} | DLoss:{:.8f} | GLoss:{:.8f}".
            format(
                NowStr,
                self.CurrEpochIndex,
                self.CurrBatchIndex,
                self.CurrBatchDiscriminatorLoss,
                self.CurrBatchGeneratorLoss
            )
        )
        pass

###########################################################################################
