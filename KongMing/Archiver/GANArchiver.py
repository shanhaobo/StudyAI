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
        super().__init__(inModelPrefix, inModelRootFolderPath)
        self.Generator = inGenerator
        self.Discriminator = inDiscriminator

        self.NNModelDict["Generator"] = self.Generator
        self.NNModelDict["Discriminator"] = self.Discriminator
        
        self.NNModelNameOnlyForEval.append("Generator")
