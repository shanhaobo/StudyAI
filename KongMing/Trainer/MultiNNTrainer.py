from .BaseTrainer import BaseTrainer

import torch
from typing import Dict as TypedDict

class MultiNNTrainer(BaseTrainer) :
    def __init__(
            self,
            inLearningRate
        ) -> None:
        super().__init__(
            inLearningRate
        )
        self.NNModuleDict : TypedDict[str, torch.nn.Module] = {}

    def RegisterMultiNNModule(
            self,
            inNNModelDict : TypedDict[str, torch.nn.Module]
        ) -> None:
        self.NNModuleDict = inNNModelDict
