import torch

from .MultiNNArchiver import MultiNNArchiver

class DDPMArchiver(MultiNNArchiver):
    def __init__(
            self,
            inNNModel : torch.nn.Module,
            inDiffusionModel : torch.nn.Module,
            inModelRootFolderPath: str
        ) -> None:
        super().__init__(
            {"NNModel" : inNNModel, "DiffusionModel" : inDiffusionModel},
            inModelRootFolderPath
        )

        self.NNModelNameOnlyForTrain.append("NNModel")
