from .BaseArchiver import BaseArchiver

import torch
from typing import Dict as TypedDict
from typing import List as TypedList

class MultiNNArchiver(BaseArchiver):
    def __init__(
            self,
            inModelRootFolderPath : str,
            inNNModuleNameOnlyForTrain : TypedList[str] = []
        ) -> None:
        super().__init__(inModelRootFolderPath, inNNModuleNameOnlyForTrain)

    def RegisterMultiNNModule(
            self,
            inNNModelDict : TypedDict[str, torch.nn.Module]
        ) -> None:
        self.NNModuleDict = inNNModelDict
