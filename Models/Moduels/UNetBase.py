import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBase(nn.Module):
    def __init__(
            self,
            inInputDim,
            inOutputDim,
            inImageSize,
            inLevelCountORList,
            InputModuleType,
            DownsampleModuleType,
            MidModuleType,
            UpsampleModuleType,
            OutputModuleType
        ) -> None:
        super().__init__()

        self.InputDim               = inInputDim
        self.OutputDim              = inOutputDim
        self.ImageSize              = inImageSize

        self.DownsampleModuleType   = DownsampleModuleType
        self.MidModuleType          = MidModuleType
        self.UpsampleModuleType     = UpsampleModuleType

        if isinstance(inLevelCountORList, tuple) or isinstance(inLevelCountORList, list):
            AllDims                 = [*(self.ImageSize * i for i in inLevelCountORList)]
        else:
            AllDims                 = [*(self.ImageSize * (2 ** i) for i in range(0, inLevelCountORList + 1))]

        # AllDims = (1, 3, 6, 12, 24, 48, 96) 
        # list(zip(AllDims[:-1], AllDims[1:])) -> ((1, 3, 6, 12, 24, 48), (3, 6, 12, 24, 48, 96))
        InOutPairDims               = list(zip(AllDims[:-1], AllDims[1:]))

        # 1 -> input
        self.InputModule            = InputModuleType(self.InputDim, AllDims[0])

        self.DownSample             = nn.MaxPool2d(2)

        # 2 -> downsample
        self.DownsampleList         = nn.ModuleList([])
        for _, (inDim, outDim) in enumerate(InOutPairDims):
            self.DownsampleList.append(
                self.DownsampleModuleType(inDim, outDim)
            )

        # 3 -> Mid
        self.MidModule              = MidModuleType(AllDims[-1], AllDims[-1])

        # 4 -> upsample
        self.UpsampleList           = nn.ModuleList([])
        for _, (inDim, outDim) in enumerate(reversed(InOutPairDims)):
            self.UpsampleList.append(
                self.UpsampleModuleType(outDim * 2, inDim)
            )

        self.UpSample               = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 5 -> output
        self.OutputModule           = OutputModuleType(AllDims[0], self.OutputDim)

    def forward(self, inData):

        Stack = []

        # 1
        X = self.InputModule(inData)

        # 2
        for DM in self.DownsampleList:
            X = self.DownSample(X)
            X = DM(X)
            Stack.append(X)

        # 3
        X = self.MidModule(X)
        
        # 4
        for UM in self.UpsampleList:
            # dim=1 是除Batch后的一个维度,这个维度很可能是Channels
            X = torch.cat((X, Stack.pop()), dim=1)
            X = UM(X)
            X = self.UpSample(X)

        # 5
        return self.OutputModule(X)

"""
x:torch.Size([3, 4])  
y:torch.Size([3, 4])

cat dim0 torch.Size([6, 4])
cat dim1 torch.Size([3, 8])
"""
