import torch
import torch.nn as nn
import torch.optim as optim

from .MultiNNTrainer import MultiNNTrainer

class GANTrainer(MultiNNTrainer):
    def __init__(
            self,
            inGenerator : nn.Module,
            inDiscriminator : nn.Module,
            inGeneratorInputSize,
            inLearningRate = 1e-5,
        ) -> None:
        super().__init__(inLearningRate)
        
        self.Generator          = inGenerator.to(self.Device)
        self.Discriminator      = inDiscriminator.to(self.Device)

        self.GeneratorInputSize = inGeneratorInputSize
        pass

    def _CreateOptimizer(self) -> None:
        self.OptimizerG = optim.Adam(self.Generator.parameters(),     lr=self.LearningRate, betas=(0.5, 0.999))
        self.OptimizerD = optim.Adam(self.Discriminator.parameters(), lr=self.LearningRate, betas=(0.5, 0.999))
        pass

    def _CreateLossFN(self) -> None:
        self.LossFN     = nn.BCELoss()
        pass

    def _CalcLossForReal(self, inBatchData):
        DiscriminatorResult = self.Discriminator(inBatchData)
        RealLabels = torch.ones(DiscriminatorResult.size()).to(self.Device)
        return self.LossFN(DiscriminatorResult, RealLabels)
    
    def _CalcLossForFake(self, inBatchData):
        DiscriminatorResult = self.Discriminator(inBatchData)
        FakeLabels = torch.zeros(DiscriminatorResult.size()).to(self.Device)
        return self.LossFN(DiscriminatorResult, FakeLabels)

    def _BeginBatchTrain(self, inBatchIndex, **inArgs) -> None:
        pass

    def _BatchTrain(self, inBatchIndex, inBatchData, **inArgs) :
        self._BeginBatchTrain(inBatchIndex, **inArgs)

        nBatchSize = inBatchData.size(0)
        BatchGeneratorInputSize = (nBatchSize, ) + self.GeneratorInputSize
        
        # Optimize Discriminator
        RealBatchData = inBatchData.to(self.Device)
        DLossReal = self._CalcLossForReal(RealBatchData)

        FakeBatchData = self.Generator(torch.randn(BatchGeneratorInputSize).to(self.Device))
        DLossFake = self._CalcLossForFake(FakeBatchData)

        DLoss = (DLossReal + DLossFake) * 0.5
        self._BackPropagate(self.OptimizerD, DLoss)
        
        # Optimize Generator
        FakeBatchData = self.Generator(torch.randn(BatchGeneratorInputSize).to(self.Device))
        GLoss = self._CalcLossForReal(FakeBatchData)
        self._BackPropagate(self.OptimizerG, GLoss)

        self._EndBatchTrain(inBatchIndex, CurrDLoss = DLoss.item(), CurrGLoss = GLoss.item(), **inArgs)
    
    def _EndBatchTrain(self, inBatchIndex, **inArgs) -> None:
        pass

    def _BeginEpochTrain(self, inEpochIndex, **inArgs) -> None:
        pass

    def _EndEpochTrain(self, inEpochIndex, **inArgs) -> None:
        pass

    def _BeginTrain(self, **inArgs) -> None:
        pass

    def _EndTrain(self, **inArgs) -> None:
        pass
