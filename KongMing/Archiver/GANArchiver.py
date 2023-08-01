import torch

from .MultiNNArchiver import MultiNNArchiver

class GANArchiver(MultiNNArchiver):
    def __init__(
            self,
            inGenerator : torch.nn.Module,
            inDiscriminator : torch.nn.Module,
            inModelRootFolderPath : str = "."
        ) -> None:
        super().__init__(
            {"Generator" : inGenerator, "Discriminator" : inDiscriminator},
            inModelRootFolderPath
        )

        self.NNModuleNameOnlyForTrain.append("Discriminator")
