import torch

from KongMing.Utils.CaseInsensitiveContainer import CaseInsensitiveList, CaseInsensitiveDict

from KongMing.ModelFactory.MultiNNModelFactory import MultiNNModelFacotry

from KongMing.Trainer.GANs.GANTrainer import GANTrainer
from KongMing.Trainer.GANs.WGANTrainer import WGANTrainer

from KongMing.Archiver.MultiNNArchiver import MultiNNArchiver

class GANModelFactory(MultiNNModelFacotry):
    def __init__(
            self,
            inGenerator : torch.nn.Module,
            inDiscriminator : torch.nn.Module,
            inGeneratorEmbeddingDim,
            inWTrainer = True,
            inLearningRate = 1e-5,
            inModelRootFolderPath = "."
        ) -> None:
        
        if inWTrainer :
            NewTrainer = WGANTrainer(
                inGeneratorEmbeddingDim,
                inLearningRate
            )
        else:
            NewTrainer = GANTrainer(
                inGeneratorEmbeddingDim,
                inLearningRate
            )

        NewArchiver = MultiNNArchiver(
            inModelRootFolderPath,
            inNNModuleNameOnlyForTrain = ["Discriminator"]
        )

        super().__init__({"Generator" : inGenerator, "Discriminator" : inDiscriminator}, NewTrainer, NewArchiver)

        self.GeneratorEmbeddingDim = inGeneratorEmbeddingDim

        g = self._SumParameters(inGenerator)
        d = self._SumParameters(inDiscriminator)
        print("Sum of Params:{:,} | Generator Params:{:,} | Discriminator Params:{:,}".format(g + d, g, d))

    ###########################################################################################

    def Eval(self, inEpoch, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None):
        if (super().Eval(inEpoch, inArgs, inKVArgs) == False) :
            return None
        
        BatchSize = inKVArgs.get("inBatchSize")
        if (BatchSize is None) :
            BatchSize = 1
        ImageSize=inKVArgs["inImageSize"]
        ColorChanNum=inKVArgs["inColorChanNum"]

        self.Trainer.Generator.eval()
        return self.Trainer.Generator(torch.randn((BatchSize, self.GeneratorEmbeddingDim, ImageSize, ImageSize)).to(self.Trainer.Device))

    ###########################################################################################
