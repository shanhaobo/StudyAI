import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBase(nn.Module):
    def __init__(self, inInputDim, inOutputDim, inImageSize, inLevelCount, DownsampleModuleType, UpsampleModuleType, MidModuleType) -> None:
        super().__init__()

        self.InputDim               = inInputDim
        self.OutputDim              = inOutputDim
        self.ImageSize              = inImageSize

        self.DownsampleModuleType   = DownsampleModuleType
        self.UpsampleModuleType     = UpsampleModuleType
        self.MidModuleType          = MidModuleType

        AllDims = [self.InputDim, *(self.ImageSize * i for i in range(1, inLevelCount))]
        InOutPairDims = list(zip(AllDims[:-1], AllDims[1:]))

        self.DownsampleList         = nn.ModuleList([])
        self.UpsampleList           = nn.ModuleList([])

        for _, (inDim, outDim) in enumerate(InOutPairDims):
            self.DownsampleList.append(
                self.DownsampleModuleType(inDim, outDim)
            )

        self.MidModule = MidModuleType(AllDims[-1], AllDims[-1])

        for _, (inDim, outDim) in enumerate(reversed(InOutPairDims)):
            self.UpsampleList.append(
                self.UpsampleModuleType(inDim, outDim)
            )

    def forward(self, inData):
        pass


