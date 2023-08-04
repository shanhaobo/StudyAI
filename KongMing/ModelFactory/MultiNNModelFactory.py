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
        NewArchiver = MultiNNArchiver(
            inModelRootFolderPath,
            inNNModuleNameOnlyForTrain
        )
        # set Log Root Path
        inTrainer.LogRootPath = NewArchiver.GetCurrTrainRootPath()

        super().__init__(inTrainer, NewArchiver)

        for Name, NN in inMultiNNDict.items():
            self.MultiNNDict[Name] = NN.to(self.Device)

        self.Trainer.RegisterMultiNNModule(self.MultiNNDict)
        self.Archiver.RegisterMultiNNModule(self.MultiNNDict)
