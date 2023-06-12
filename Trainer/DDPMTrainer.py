import torch

from .BaseTrainer import BaseTrainer

class DDPMTrainer(BaseTrainer) :
    def __init__(self, 
            inDDPM : torch.nn.Module,
            inLearningRate,
            inTimesteps = 1000
        ) -> None:
        super().__init__(inLearningRate)
        self.DiffusionModel = inDDPM

        self.Timesteps = inTimesteps

        pass

    def Initialize(self):
        
        pass

    def _CreateOptimizer(self) -> None:
        self.Optimizer = torch.optim.Adam(self.DiffusionModel.parameters(),     lr=self.LearningRate, betas=(0.5, 0.999))
        pass

    def _CreateLossFN(self) -> None:
        pass

    def _BatchTrain(self, inBatchDatum, inBatchLabel, *inArgs, **inKWArgs) :
        # get BatchSize
        nBatchSize = inBatchDatum.size(0)
        
        pass
