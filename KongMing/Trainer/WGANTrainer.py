import torch
import torch.nn as nn

from .GANTrainer import GANTrainer
from torch.autograd import grad
import torch.nn.functional as F

class WGANTrainer(GANTrainer) :
    def __init__(
            self,
            inGenerator: nn.Module,
            inDiscriminator: nn.Module,
            inGeneratorInputSize,
            inLearningRate=0.00001,
            inLogRootPath="."
        ) -> None:
        super().__init__(inGenerator, inDiscriminator, inGeneratorInputSize, inLearningRate, inLogRootPath)

    def _CreateOptimizer(self) -> None:
        self.OptimizerG = torch.optim.Adam(self.Generator.parameters(),     lr=self.LearningRate, betas=(0.5, 0.999))
        self.OptimizerD = torch.optim.Adam(self.Discriminator.parameters(), lr=self.LearningRate, betas=(0.5, 0.999))
        pass

    def _CreateLossFN(self) -> None:
        pass

    def __CalculateGradientPenalty(self, RealData, FakeData):
        FakeData = F.interpolate(FakeData, size=(RealData.shape[2], RealData.shape[3]), mode='bilinear', align_corners=False)
        alpha = torch.rand(RealData.size(0), 1, 1, 1).to(self.Device)
        interpolates = (alpha * RealData + ((1 - alpha) * FakeData)).requires_grad_(True)
        d_interpolates = self.Discriminator(interpolates)
        gradients = grad(outputs=d_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(d_interpolates.size()).to(self.Device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def _CalcDiscriminatorLoss(self, RealData, RealScores, FakeData, FakeScores):
        # Calc W Loss
        GradientPenalty = self.__CalculateGradientPenalty(RealData, FakeData.detach())
        return -(torch.mean(RealScores) - torch.mean(FakeScores)) + GradientPenalty
    
    def _CalcGeneratorLoss(self):
        return 

    
    def _BatchTrain(self, inBatchData, inBatchLabel, *inArgs, **inKWArgs) :
        # get BatchSize
        nBatchSize = inBatchData.size(0)
        # Prepare Real and Fake Data
        RealData = inBatchData.to(self.Device)
        FakeData = self.Generator(torch.randn((nBatchSize,) + self.GeneratorInputSize, device=self.Device))
        print("DataSize:{} {}".format(RealData.size(),FakeData.size()))

        # Calc Scores
        RealScores                  = self.Discriminator(RealData)
        FakeScoresForDiscriminator  = self.Discriminator(FakeData.detach())

        # Calc W Loss
        DLoss                       = self._CalcDiscriminatorLoss(RealData, RealScores, FakeData, FakeScoresForDiscriminator)
        # Optimize Discriminator
        self._BackPropagate(self.OptimizerD, DLoss)
        
        # Optimize Generator
        FakeScoresForGenerator      = self.Discriminator(FakeData)
        GLoss                       = -torch.mean(FakeScoresForGenerator)
        self._BackPropagate(self.OptimizerG, GLoss)

        self.CurrBatchDiscriminatorLoss = DLoss.item()
        self.CurrBatchGeneratorLoss     = GLoss.item()

        pass

