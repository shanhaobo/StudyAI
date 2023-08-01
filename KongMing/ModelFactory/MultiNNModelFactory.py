import torch

from KongMing.Archiver.MultiNNArchiver import MultiNNArchiver
from KongMing.Trainer.MultiNNTrainer import MultiNNTrainer
from .BaseModelFactory import BaseModelFactory

from typing import Dict as TypedDict

class MultiNNModelFacotry(BaseModelFactory):
    def __init__(
            self,
            inMultiNNDict : TypedDict[str, torch.nn.Module],
            inTrainer : MultiNNTrainer,
            inArchiver : MultiNNArchiver
        ):
        self.MultiNNDict : TypedDict[str, torch.nn.Module] = {}
        
        super().__init__(inTrainer, inArchiver)

        for Name, NN in inMultiNNDict.items():
            self.MultiNNDict[Name] = NN.to(self.Device)

        self.Trainer.RegisterMultiNNModule(self.MultiNNDict)
        self.Archiver.RegisterMultiNNModule(self.MultiNNDict)
