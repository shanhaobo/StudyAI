import torch

from .BaseArchiver import BaseArchiver

class SingleNNArchiver(BaseArchiver):
    def __init__(
            self,
            inModelRootFolderPath : str,
            inModelTag : str = None
        ) -> None:
        super().__init__(inModelRootFolderPath, inModelTag=inModelTag)

        self.NNModuleDict["NNModel"] = None
