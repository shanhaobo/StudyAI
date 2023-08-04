import torch
from .BaseModelFactory import BaseModelFactory

from KongMing.Archiver.MultiNNArchiver import MultiNNArchiver
from KongMing.Trainer.MultiNNTrainer import MultiNNTrainer

from typing import Dict as TypedDict
from typing import List as TypedList

class MultiNNModelFacotry(BaseModelFactory):
    def __init__(
            self,
            inMultiNNDict : TypedDict[str, torch.nn.Module],
            inTrainer : MultiNNTrainer,
            inModelRootFolderPath : str,
            inNNModuleNameOnlyForTrain : TypedList[str] = []
        ):
        self.MultiNNDict : TypedDict[str, torch.nn.Module] = {}

        # new Archiver
        Archiver = MultiNNArchiver(
            inModelRootFolderPath,
            inNNModuleNameOnlyForTrain
        )
        # set Log Root Path
        inTrainer.LogRootPath = Archiver.GetCurrTrainRootPath()

        super().__init__(inTrainer, Archiver)

        for Name, NN in inMultiNNDict.items():
            self.MultiNNDict[Name] = NN.to(self.Device)

        self.Trainer.RegisterMultiNNModule(self.MultiNNDict)
        self.Archiver.RegisterMultiNNModule(self.MultiNNDict)
