
from Models.BaseModel import BaseModel

from Archiver.DDPMArchiver import DDPMArchiver
from Trainer.DDPMTrainer import DDPMTrainer

from Modules.UNet2D import UNet2D, UNet2DPosEmbed, UNet2DAttnPosEmbed

from .DiffusionModelUtils import ConditionUNet

from .DiffusionModelBase import DiffusionModel

class DDPMModel(BaseModel) :
    def __init__(self, inEmbedDim, inChannel, inLearningRate=0.00001, inTimesteps : int = 1000, inModeRootlFolderPath="."):
        self.NNModel        = UNet2DAttnPosEmbed(inChannel=inChannel, inEmbedDim=inEmbedDim, inEmbedLvlCntORList=3)
        #self.NNModel        = ConditionUNet(inChannel=inChannel, inEmbedDim=inEmbedDim, inEmbedLvlCntORList=(1, 2, 4))
        self.DiffusionModel = DiffusionModel(inTimesteps=inTimesteps)
        NewTrainer          = DDPMTrainer(self.NNModel, self.DiffusionModel, inLearningRate, inTimesteps=inTimesteps)
        NewArchiver         = DDPMArchiver(self.NNModel, self.DiffusionModel, inModeRootlFolderPath)
        super().__init__(NewTrainer, NewArchiver)

        m = self._SumParameters(self.NNModel)
        b = self._SumParameters(self.DiffusionModel)
        print("Sum of Params:{} | Model Params:{} | Buffer Params:{}".format(m + b, m, b))


    def Eval(self, *inArgs, **inKWArgs):
        if (super().Eval(*inArgs, **inKWArgs) == False) :
            return None
        self.DiffusionModel.eval()
        self.NNModel.eval()
        return self.DiffusionModel.Sample(
            self.NNModel,
            inImageSize=inKWArgs["inImageSize"],
            inImageChannel=inKWArgs["inImageChannel"],
            inBatchSize=inKWArgs["inBatchSize"]
        )
