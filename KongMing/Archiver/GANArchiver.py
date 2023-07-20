import torch

from .MultiNNArchiver import MultiNNArchiver

class GANArchiver(MultiNNArchiver):
    def __init__(
            self,
            inGenerator : torch.nn.Module,
            inDiscriminator : torch.nn.Module,
            inModelPrefix : str = "GAN",
            inModelRootFolderPath : str = "."
        ) -> None:
        super().__init__(
            {"Generator" : inGenerator, "Discriminator" : inDiscriminator},
            inModelPrefix,
            inModelRootFolderPath
        )

        self.NNModelNameOnlyForTrain.append("Discriminator")
