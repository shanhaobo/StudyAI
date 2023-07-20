import torch

from .BaseArchiver import BaseArchiver

class SingleNNArchiver(BaseArchiver):
    def __init__(
            self, 
            inNNModel : torch.nn.Module,
            inModelPrefix : str, 
            inModelRootFolderPath : str
        ) -> None:
        super().__init__(inModelPrefix, inModelRootFolderPath)

        self.NNModelDict[inModelPrefix] = inNNModel
