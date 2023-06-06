import torch
import torch.nn as nn

from .GANTrainer import GANTrainer
from torch.autograd import grad
import torch.nn.functional as F

class WGANTrainer(GANTrainer) :
    def __init__(self, inGenerator: nn.Module, inDiscriminator: nn.Module, inGeneratorInputSize, inLearningRate=0.00001) -> None:
        super().__init__(inGenerator, inDiscriminator, inGeneratorInputSize, inLearningRate)

    def calculate_gradient_penalty(self, real_images, fake_images):
        alpha = torch.rand(real_images.size(0), 1, 1, 1).to(self.Device)
        interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)
        d_interpolates = self.Discriminator(interpolates)
        gradients = grad(outputs=d_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(d_interpolates.size()).to(self.Device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    
    def _BatchTrain(self, inBatchDatum, inBatchLabel, *inArgs, **inKWArgs) :
        # get BatchSize
        nBatchSize = inBatchDatum.size(0)
        
        # Optimize Discriminator
        RealDatum = inBatchDatum.to(self.Device)
        GenFakeDatum = self.Generator(torch.randn((nBatchSize,) + self.GeneratorInputSize, device=self.Device))
        GenFakeDatum = F.interpolate(GenFakeDatum, size=(RealDatum.shape[2], RealDatum.shape[3]), mode='bilinear', align_corners=False)
        #print("DatumSize:{} {}".format(RealDatum.size(),GenFakeDatum.size()))

        RealScores = self.Discriminator(RealDatum)
        FakeScores = self.Discriminator(GenFakeDatum.detach())

        gradient_penalty = self.calculate_gradient_penalty(RealDatum, GenFakeDatum.detach())

        DLoss = -(torch.mean(RealScores) - torch.mean(FakeScores)) + gradient_penalty
        self._BackPropagate(self.OptimizerD, DLoss)
        
        # Optimize Generator
        fake_scores = self.Discriminator(GenFakeDatum)
        GLoss = -torch.mean(fake_scores)
        self._BackPropagate(self.OptimizerG, GLoss)

        self.CurrBatchDiscriminatorLoss = DLoss.item()
        self.CurrBatchGeneratorLoss     = GLoss.item()

        pass

    #d_loss = -(torch.mean(real_scores) - torch.mean(fake_scores)) + gradient_penalty

