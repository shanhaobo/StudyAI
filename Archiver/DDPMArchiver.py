import torch

from .BaseArchiver import BaseArchiver

class DDPMArchiver(BaseArchiver):
    def __init__(
            self,
            inModelPrefix: str = "DDPM",
            inModelRootFolderPath: str = "."
        ) -> None:
        super().__init__(inModelPrefix, inModelRootFolderPath)
        
