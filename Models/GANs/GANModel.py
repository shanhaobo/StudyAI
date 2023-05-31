import torch
from torch.utils.data import DataLoader

from datetime import datetime

from Models.BaseModel import BaseModel

from Trainer.GANTrainer import GANTrainer
from Utils.Archiver import GANArchiver

class GANModel(BaseModel):
    def __init__(
            self,
            inGenerator : torch.nn.Module,
            inDiscriminator : torch.nn.Module,
            inGeneratorSize,
            inLearningRate = 1e-5,
            inModelRootFolderPath = "."
        ) -> None:
        
        NewTrainer = GANTrainer(
            inGenerator,
            inDiscriminator,
            inGeneratorSize,
            inLearningRate
        )

        NewArchiver = GANArchiver(
            inGenerator,
            inDiscriminator,
            inModelRootFolderPath
        )

        super().__init__(NewTrainer, NewArchiver)

        self.Trainer.EndBatchTrain.add(self.EndBatchTrain)
        self.Trainer.EndEpochTrain.add(self.EndEpochTrain)
        self.Trainer.EndTrain.add(self.EndTrain)
        self.Trainer.BeginTrain.add(self.BeginTrain)

    def Train(self, inDataLoader : DataLoader, inNumEpochs : int = 0, *inArgs, **inKWArgs) -> None:
        super().Train(inDataLoader, inNumEpochs, *inArgs, **inKWArgs)

    def IncTrain(self, inDataLoader : DataLoader, inNumEpochs : int = 0, *inArgs, **inKWArgs) -> None:
        super().IncTrain(inDataLoader, inNumEpochs, *inArgs, **inKWArgs)

    def Eval(self, *inArgs, **inKWArgs):
        if (super().Eval(*inArgs, **inKWArgs) == False) :
            return None
        self.Trainer.Generator.eval()
        return self.Trainer.Generator(torch.randn((1, ) + self.Trainer.GeneratorInputSize).to(self.Trainer.Device))

    def IsExistModels(self, inForTrain : bool = True, *inArgs, **inKWArgs) -> bool:
        return self.Archiver.IsExistModel(inForTrain, *inArgs, **inKWArgs)
    
    def EndBatchTrain(self, *inArgs, **inKWArgs) -> None:
        NowStr  = datetime.now().strftime("%Y%m%d:%H%M%S:%f")
        print(
            "{} | Epoch:{:0>4d} | Batch:{:0>6d} | DLoss:{:.8f} | GLoss:{:.8f}".
            format(
                NowStr,
                self.Trainer.CurrEpochIndex + 1,
                self.Trainer.CurrBatchIndex + 1,
                self.Trainer.CurrBatchDiscriminatorLoss,
                self.Trainer.CurrBatchGeneratorLoss
            )
        )
        pass

    def EndEpochTrain(self, *inArgs, **inKWArgs) -> None:
        interval = inKWArgs["SaveModelInterval"]
        if self.Trainer.CurrEpochIndex % interval == 0 and self.Trainer.CurrEpochIndex > 0 :
            print("Epoch:{} Save Models".format(self.Trainer.CurrEpochIndex + 1))
            self.Archiver.Save("_{:0>4d}".format(self.Trainer.CurrEpochIndex + 1))
        pass

    def EndTrain(self, *inArgs, **inKWArgs)->None:
        self.Archiver.Save("")
        print("End Train")

    def BeginTrain(self, *inArgs, **inKWArgs)->None:
        print("Begin Training...")
