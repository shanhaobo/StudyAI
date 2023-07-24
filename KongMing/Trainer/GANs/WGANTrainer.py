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
        #self.Generator.ApplyOptimizer(torch.optim.RMSprop, self.LearningRate)
        #self.Discriminator.ApplyOptimizer(torch.optim.RMSprop, self.LearningRate)
        self.Generator.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))
        self.Discriminator.ApplyOptimizer(torch.optim.Adam, self.LearningRate, betas=(0.5, 0.999))

    def _CreateLossFN(self) -> None:
        pass

    # 计算梯度惩罚
    def __CalculateGradientPenalty(self, RealData, FakeData):
        BatchSize, _, ImageHeight, ImageWidth = RealData.size()

        # 将数据插值成与RealData的尺寸一致, 这里其实可以不做,因为我已经做了
        FakeData = F.interpolate(FakeData, size=(ImageHeight, ImageWidth), mode='bilinear', align_corners=False)
        # *size (Batch, 1, 1, 1)
        Alpha = torch.rand(BatchSize, 1, 1, 1, device=self.Device)
        # 计算插值
        Interpolates = (Alpha * RealData + ((1 - Alpha) * FakeData)).requires_grad_(True)
        # 计数插值分数
        InterpolatesScores = self.Discriminator(Interpolates)
        # 计算InterpolatesScores对Interpolates的梯度
        Gradients = grad(outputs=InterpolatesScores, inputs=Interpolates,
                        grad_outputs=torch.ones(InterpolatesScores.size(), device=self.Device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        # 计算梯度惩罚
        GradientPenalty = ((Gradients.norm(2, dim=1) - 1) ** 2).mean()

        return GradientPenalty
    
    def _CalcDiscriminatorLoss(self, RealData, RealScores, FakeData, FakeScores):
        # Calc W Loss
        GradientPenalty = self.__CalculateGradientPenalty(RealData, FakeData.detach())

        return -(torch.mean(RealScores) - torch.mean(FakeScores)) + GradientPenalty
    
    def _CalcGeneratorLoss(self, FakeData):
        FakeScoresForGenerator      = self.Discriminator(FakeData)
        return -torch.mean(FakeScoresForGenerator) 

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :

        BatchSize, _, ImageHeight, ImageWidth = inBatchData.size()

        # Prepare Real and Fake Data
        RealData = inBatchData.to(self.Device)

        with self.Discriminator as D:
            with self.Generator as G:
                FakeData = self.Generator(
                    torch.randn(
                        (BatchSize, self.GeneratorEmbeddingDim, ImageHeight, ImageWidth),
                        device=self.Device
                    )
                )
                GLoss    = self._CalcGeneratorLoss(FakeData)
                G.AcceptLoss(GLoss)

            # Calc Scores
            RealScores  = self.Discriminator(RealData)
            FakeScores  = self.Discriminator(FakeData.detach())

            # Calc W Loss
            DLoss       = self._CalcDiscriminatorLoss(RealData, RealScores, FakeData, FakeScores)
            D.AcceptLoss(DLoss)
