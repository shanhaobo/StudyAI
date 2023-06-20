
from Models.BaseModel import BaseModel

from Archiver.DDPMArchiver import DDPMArchiver
from Trainer.DDPMTrainer import DDPMTrainer

from ..Moduels.UNet2D import UNet2D

from .DiffusionModelUtils import ConditionUNet

from .DiffusionModelBase import DiffusionModel

class DDPMModel(BaseModel) :
    def __init__(self, inImageSize, inChannel, inLearningRate=0.00001, inTimesteps : int = 1000, inModeRootlFolderPath="."):
        self.NNModel        = ConditionUNet(dim=inImageSize, channels=inChannel, dim_mults=(1,2,4,))
        self.DiffusionModel = DiffusionModel(inTimesteps=inTimesteps)
        NewTrainer          = DDPMTrainer(self.NNModel, self.DiffusionModel, inLearningRate, inTimesteps=inTimesteps)
        NewArchiver         = DDPMArchiver(self.NNModel, self.DiffusionModel, inModeRootlFolderPath)
        super().__init__(NewTrainer, NewArchiver)

    def Eval(self, *inArgs, **inKWArgs):
        if (super().Eval(*inArgs, **inKWArgs) == False) :
            return None
        self.DiffusionModel.eval()
        self.NNModel.eval()
        return self.DiffusionModel.Sample(
            self.NNModel,
            inImageSize=inKWArgs["inImageSize"],
            inBatchSize=inKWArgs["inBatchSize"],
            inChannels=inKWArgs["inChannels"]
        )
