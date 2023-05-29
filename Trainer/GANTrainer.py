import torch
import torch.nn as nn
import torch.optim as optim

from functools import partial

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
        
        self.Generator                  = inGenerator.to(self.Device)
        self.Discriminator              = inDiscriminator.to(self.Device)

        self.GeneratorInputSize         = inGeneratorInputSize

        pass

    def _CreateOptimizer(self) -> None:
        self.OptimizerG = optim.Adam(self.Generator.parameters(),     lr=self.LearningRate, betas=(0.5, 0.999))
        self.OptimizerD = optim.Adam(self.Discriminator.parameters(), lr=self.LearningRate, betas=(0.5, 0.999))
        pass

    def _CreateLossFN(self) -> None:
        self.LossFN     = nn.BCELoss().to(self.Device)
        pass

    def _CalcLossForReal(self, inBatchData):
        DiscriminatorResult = self.Discriminator(inBatchData)
        RealLabels = torch.ones(DiscriminatorResult.size()).to(self.Device)
        return self.LossFN(DiscriminatorResult, RealLabels)
    
    def _CalcLossForFake(self, inBatchData):
        DiscriminatorResult = self.Discriminator(inBatchData)
        FakeLabels = torch.zeros(DiscriminatorResult.size()).to(self.Device)
        return self.LossFN(DiscriminatorResult, FakeLabels)


    def _BatchTrain(self, inBatchDatum, inBatchLabel, *inArgs, **inKWArgs) :

        nBatchSize = inBatchDatum.size(0)
        BatchGeneratorInputSize = (nBatchSize, ) + self.GeneratorInputSize
        
        # Optimize Discriminator
        RealDatum = inBatchDatum.to(self.Device)
        DLossReal = self._CalcLossForReal(RealDatum)

        GenFakeDatum = self.Generator(torch.randn(BatchGeneratorInputSize).to(self.Device))
        # detach() do not effect Generator
        DLossFake = self._CalcLossForFake(GenFakeDatum.detach())

        DLoss = (DLossReal + DLossFake) * 0.5
        self._BackPropagate(self.OptimizerD, DLoss)
        
        # Optimize Generator
        GLoss = self._CalcLossForReal(GenFakeDatum) 
        self._BackPropagate(self.OptimizerG, GLoss)

        self.CurrBatchDiscriminatorLoss = DLoss.item()
        self.CurrBatchGeneratorLoss     = GLoss.item()

        pass
