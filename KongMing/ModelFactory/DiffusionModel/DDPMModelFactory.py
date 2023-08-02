import torch

from ..MultiNNModelFactory import MultiNNModelFacotry

from KongMing.Archiver.MultiNNArchiver import MultiNNArchiver
from KongMing.Trainer.DDPMTrainer import DDPMTrainer

from KongMing.Utils.CaseInsensitiveContainer import CaseInsensitiveList, CaseInsensitiveDict

from .DiffusionModelBase import DiffusionModel
from .UNet2D import UNet2D_ConvNeXt, UNet2D_WSR

class DDPMModelFactory(MultiNNModelFacotry) :
    def __init__(
            self,
            inEmbeddingDim,
            inColorChanNum,
            inLearningRate=0.00001,
            inTimesteps : int = 1000,
            inModelRootFolderPath="."
        ):
        self.NNModel =  UNet2D_WSR(inColorChanNum=inColorChanNum, inEmbeddingDim=inEmbeddingDim, inEmbedLvlCntORList=(1, 2, 4))
        #self.NNModel = UNet2D_ConvNeXt(inColorChanNum=inColorChanNum, inEmbeddingDim=inEmbeddingDim, inEmbedLvlCntORList=(1, 2, 4))
        self.DiffusionModel = DiffusionModel(inTimesteps=inTimesteps, inNNModule=self.NNModel)

        NewArchiver         = MultiNNArchiver(
                                inModelRootFolderPath,
                                inNNModuleNameOnlyForTrain = ["NNModel"]
                            )
        NewTrainer          = DDPMTrainer(
                                inLearningRate,
                                inLogRootPath=NewArchiver.GetCurrTrainRootPath()
                            )
        super().__init__({"NNModel" : self.NNModel, "DiffusionModel" : self.DiffusionModel}, NewTrainer, NewArchiver)

        m = self._SumParameters(self.NNModel)
        b = self._SumParameters(self.DiffusionModel)
        print("Sum of Params:{:,} | Model Params:{:,} | Buffer Params:{:,}".format(m + b, m, b))

    def Eval(self, inEpoch, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None):
        if (super().Eval(inEpoch, inArgs, inKVArgs) == False) :
            return None
        self.DiffusionModel.eval()
        return self.DiffusionModel.Sample(
            self.DiffusionModel.EMA,
            inImageSize=inKVArgs["inImageSize"],
            inColorChanNum=inKVArgs["inColorChanNum"],
            inBatchSize=inKVArgs["inBatchSize"]
        )
