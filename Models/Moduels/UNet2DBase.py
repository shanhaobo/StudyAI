import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################################
######################################################################################
######################################################################################

class UNet2DBase(nn.Module):
    def __init__(
            self,
            inInputDim,
            inOutputDim,
            inEmbedDims,
            inEmbedLvlCntORList,
            InputModuleType,
            DownsampleModuleType,
            MidModuleType,
            UpsampleModuleType,
            OutputModuleType
        ) -> None:
        super().__init__()

        self.InputDim               = inInputDim
        self.OutputDim              = inOutputDim
        self.EmbedDim               = inEmbedDims

        self.DownsampleModuleType   = DownsampleModuleType
        self.MidModuleType          = MidModuleType
        self.UpsampleModuleType     = UpsampleModuleType

        if isinstance(inEmbedLvlCntORList, tuple) or isinstance(inEmbedLvlCntORList, list):
            AllDims                 = [*(self.EmbedDim * i for i in inEmbedLvlCntORList)]
        else:
            AllDims                 = [*(self.EmbedDim * (2 ** i) for i in range(0, inEmbedLvlCntORList + 1))]

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
            # dim=1 是除Batch后的一个维度,这个维度很可能是EmbedDim
            X = torch.cat((X, Stack.pop()), dim=1)
            X = UM(X)
            X = self.UpSample(X)

        # 5
        return self.OutputModule(X)

######################################################################################
######################################################################################
######################################################################################

class UNet2DBaseWithExtraData(nn.Module):
    def __init__(
            self,
            inInputDim,
            inOutputDim,
            inEmbedDims,
            inEmbedLvlCntORList,
            InputModuleType,
            DownsampleModuleType,
            MidModuleType,
            UpsampleModuleType,
            OutputModuleType,
            ExtraDataProcessorType
        ) -> None:
        super().__init__()

        self.InputDim               = inInputDim
        self.OutputDim              = inOutputDim
        self.EmbedDim               = inEmbedDims

        self.DownsampleModuleType   = DownsampleModuleType
        self.MidModuleType          = MidModuleType
        self.UpsampleModuleType     = UpsampleModuleType

        if isinstance(inEmbedLvlCntORList, tuple) or isinstance(inEmbedLvlCntORList, list):
            AllDims                 = [*(self.EmbedDim * i for i in inEmbedLvlCntORList)]
        else:
            AllDims                 = [*(self.EmbedDim * (2 ** i) for i in range(0, inEmbedLvlCntORList + 1))]

        # AllDims = (1, 3, 6, 12, 24, 48, 96) 
        # list(zip(AllDims[:-1], AllDims[1:])) -> ((1, 3, 6, 12, 24, 48), (3, 6, 12, 24, 48, 96))
        InOutPairDims               = list(zip(AllDims[:-1], AllDims[1:]))

        # 1 -> input
        self.InputModule            = InputModuleType(self.InputDim, AllDims[0])

        self.DownSample             = nn.MaxPool2d(2)

        # 2 -> downsample
        self.DownsampleList         = nn.ModuleList([])
        self.DSExtraDataList        = nn.ModuleList([])
        for _, (inDim, outDim) in enumerate(InOutPairDims):
            self.DownsampleList.append(
                self.DownsampleModuleType(inDim, outDim, self.EmbedDim)
            )
            self.DSExtraDataList.append(
                nn.Sequential(nn.GELU(), nn.Linear(self.EmbedDim, inDim))
            )

        # 3 -> Mid
        self.MidModule              = MidModuleType(AllDims[-1], AllDims[-1], self.EmbedDim)
        self.MidExtraData           = nn.Sequential(nn.GELU(), nn.Linear(self.EmbedDim, AllDims[-1]))

        # 4 -> upsample
        self.UpsampleList           = nn.ModuleList([])
        self.USExtraDataList        = nn.ModuleList([])
        for _, (inDim, outDim) in enumerate(reversed(InOutPairDims)):
            self.UpsampleList.append(
                self.UpsampleModuleType(outDim * 2, inDim, self.EmbedDim)
            )
            self.USExtraDataList.append(
                nn.Sequential(nn.GELU(), nn.Linear(self.EmbedDim, outDim * 2))
            )

        self.UpSample               = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 5 -> output
        self.OutputModule           = OutputModuleType(AllDims[0], self.OutputDim)

        ### ExtraData
        self.ExtraDataProcessor     = ExtraDataProcessorType(self.EmbedDim)

    def forward(self, inData, inExtraData):

        ProcessedExtraData  = self.ExtraDataProcessor(inExtraData)

        Stack = []

        # 1
        X = self.InputModule(inData)

        # 2
        for DS, Extra in zip(self.DownsampleList, self.DSExtraDataList):
            X = self.DownSample(X)
            E = Extra(ProcessedExtraData)
            X = DS(X, E)
            Stack.append(X)

        # 3
        E = self.MidExtraData(ProcessedExtraData)
        X = self.MidModule(X, E)
        
        # 4
        for UM, Extra in zip(self.UpsampleList, self.USExtraDataList):
            # dim=1 是除Batch后的一个维度,这个维度很可能是EmbedDim
            X = torch.cat((X, Stack.pop()), dim=1)
            E = Extra(ProcessedExtraData)
            X = UM(X, E)
            X = self.UpSample(X)

        # 5
        return self.OutputModule(X)

######################################################################################
######################################################################################
######################################################################################
