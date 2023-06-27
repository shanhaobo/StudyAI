import torch

from .BaseArchiver import BaseArchiver

class MultiNNArchiver(BaseArchiver):
    def __init__(self, inModelPrefix : str, inModelRootFolderPath : str = ".") -> None:
        super().__init__(inModelPrefix, inModelRootFolderPath)

