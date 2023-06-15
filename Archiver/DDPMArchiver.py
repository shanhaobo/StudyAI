import torch

from .BaseArchiver import BaseArchiver

class DDPMArchiver(BaseArchiver):
    def __init__(
            self,
            inDDPM : torch.nn.Module,
            inModelPrefix: str = "DDPM",
            inModelRootFolderPath: str = "."
        ) -> None:
        super().__init__(inModelPrefix, inModelRootFolderPath)
        self.DDPM = inDDPM

        self.NNModelDict["DDPM"] = self.DDPM
