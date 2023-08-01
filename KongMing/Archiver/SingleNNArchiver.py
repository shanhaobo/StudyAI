import torch

from .BaseArchiver import BaseArchiver

class SingleNNArchiver(BaseArchiver):
    def __init__(
            self, 
            inNNModel : torch.nn.Module,
            inModelRootFolderPath : str
        ) -> None:
        super().__init__(inModelRootFolderPath)

        self.NNModuleDict["NNModel"] = inNNModel
