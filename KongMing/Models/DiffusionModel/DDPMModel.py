
from KongMing.Models.BaseModel import BaseModel

from KongMing.Archiver.DDPMArchiver import DDPMArchiver
from KongMing.Trainer.DDPMTrainer import DDPMTrainer

from KongMing.Modules.UNets.UNet2D import UNet2D_ConvNeXt, UNet2D_WSR
from KongMing.Modules.UNets.ConditionUNet import ConditionUNet

from .DiffusionModelBase import DiffusionModel

class DDPMModel(BaseModel) :
    def __init__(
            self,
            inEmbeddingDim,
            inColorChanNum,
            inLearningRate=0.00001,
            inTimesteps : int = 1000,
            inModeRootlFolderPath="."
        ):
        self.NNModel =  UNet2D_WSR(inColorChanNum=inColorChanNum, inEmbeddingDim=inEmbeddingDim, inEmbedLvlCntORList=(1, 2, 4))
        #self.NNModel = UNet2D_ConvNeXt(inColorChanNum=inColorChanNum, inEmbeddingDim=inEmbeddingDim, inEmbedLvlCntORList=(1, 2, 4))
        self.DiffusionModel = DiffusionModel(inTimesteps=inTimesteps)
        NewTrainer          = DDPMTrainer(self.NNModel, self.DiffusionModel, inLearningRate, inTimesteps=inTimesteps)
        NewArchiver         = DDPMArchiver(self.NNModel, self.DiffusionModel, inModeRootlFolderPath)
        super().__init__(NewTrainer, NewArchiver)

        m = self._SumParameters(self.NNModel)
        b = self._SumParameters(self.DiffusionModel)
        print("Sum of Params:{:,} | Model Params:{:,} | Buffer Params:{:,}".format(m + b, m, b))

        NewTrainer.CSVFolder= NewArchiver.CurrTrainModelArchiveRootFolderPath

    def Eval(self, *inArgs, **inKWArgs):
        if (super().Eval(*inArgs, **inKWArgs) == False) :
            return None
        self.DiffusionModel.eval()
        self.NNModel.eval()
        return self.DiffusionModel.Sample(
            self.NNModel,
            inImageSize=inKWArgs["inImageSize"],
            inColorChanNum=inKWArgs["inColorChanNum"],
            inBatchSize=inKWArgs["inBatchSize"]
        )
