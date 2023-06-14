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

    def Save(self, inEpochIndex : int) -> None:
        ModelFullPath = self.MakeNeuralNetworkArchiveFullPath("DDPM", inEpochIndex)
        torch.save(self.DDPM.state_dict(), ModelFullPath)
        print("Save Model:" + ModelFullPath)
