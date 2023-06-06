import torch

from .BaseTrainer import BaseTrainer

class DDPMTrainer(BaseTrainer) :
    def __init__(self, 
            inDDPM : torch.nn.Module,
            inLearningRate
        ) -> None:
        super().__init__(inLearningRate)
        self.DiffusionModel = inDDPM
        pass

    def _CreateOptimizer(self) -> None:
        self.Optimizer = torch.optim.Adam(self.DiffusionModel.parameters(),     lr=self.LearningRate, betas=(0.5, 0.999))
        pass

    def _CreateLossFN(self) -> None:
        self.LossFN     = torch.nn.BCELoss().to(self.Device)
        pass

    def _BatchTrain(self, inBatchDatum, inBatchLabel, *inArgs, **inKWArgs) :
        # get BatchSize
        nBatchSize = inBatchDatum.size(0)
        
        pass
