import torch

from .BaseArchiver import BaseArchiver

class DDPMArchiver(BaseArchiver):
    def __init__(
            self,
            inNNModel : torch.nn.Module,
            inDiffusionModel : torch.nn.Module,
            inModelRootFolderPath: str = "."
        ) -> None:
        super().__init__("DDPM", inModelRootFolderPath)
        self.NNModel = inNNModel
        self.DiffusionModel = inDiffusionModel

        self.NNModelDict["NNModel"] = self.NNModel
        self.NNModelDict["DiffusionModel"] = self.DiffusionModel

        self.NNModelNameOnlyForEval.append("DiffusionModel")
        
