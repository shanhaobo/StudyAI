import torch

from .BaseArchiver import BaseArchiver

class SingleNNArchiver(BaseArchiver):
    def __init__(
            self, 
            inModelRootFolderPath : str
        ) -> None:
        super().__init__(inModelRootFolderPath)

        self.NNModuleDict["NNModel"] = None
