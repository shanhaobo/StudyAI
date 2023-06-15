import torch

from datetime import datetime

from .BaseTrainer import BaseTrainer

from Models.Zoo.EMA import EMA

class DDPMTrainer(BaseTrainer) :
    def __init__(self, 
            inDDPM : torch.nn.Module,
            inLearningRate,
            inTimesteps = 1000
        ) -> None:
        super().__init__(inLearningRate)
        self.DiffusionModel = inDDPM.to(self.Device)

        self.Timesteps      = inTimesteps

        self.EMA            = EMA(inDDPM, 0.999)

        self.EndBatchTrain.add(self.DDPMEndBatchTrain)

###########################################################################################

    def _CreateOptimizer(self) -> None:
        self.Optimizer = torch.optim.Adam(self.DiffusionModel.parameters(),     lr=self.LearningRate, betas=(0.5, 0.999))
        pass

    def _CreateLossFN(self) -> None:
        pass

    def _BatchTrain(self, inBatchData, inBatchLabel, *inArgs, **inKWArgs) :
        # get BatchSize
        nBatchSize = inBatchData.size(0)
        RealData = inBatchData.to(self.Device)

        T = torch.randint(0, self.Timesteps, (nBatchSize,), device=self.Device).long()
        loss = self.DiffusionModel.P_Losses(self.DiffusionModel, RealData, T, inLossType="huber")
        
        self._BackPropagate(self.Optimizer, loss)

        self.CurrBatchDDPMLoss = loss.item()

###########################################################################################

    def DDPMEndBatchTrain(self, *inArgs, **inKWArgs) -> None:
        NowStr  = datetime.now().strftime("[%Y/%m/%d %H:%M:%S.%f]")
        print(
            "{} | Epoch:{:0>4d} | Batch:{:0>6d} | Loss:{:.8f}".
            format(
                NowStr,
                self.CurrEpochIndex + 1,
                self.CurrBatchIndex + 1,
                self.CurrBatchDDPMLoss,
            )
        )
        pass

###########################################################################################
