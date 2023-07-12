import torch
import torch.nn as nn

from einops import rearrange

from ..UtilsModules import DoubleLinearModule, DownsampleModule2D, DownsampleModule2D_PixelShuffle, UpsampleModule2D

######################################################################################
######################################################################################
######################################################################################

class UNet2DBase(nn.Module):
    def __init__(
            self,
            inInputDim,
            inOutputDim,
            inEmbeddingDim,
            inEmbedLvlCntORList,
            inInputModuleType,
            inDNSPLEncoderType,
            inMidMLMType,
            inUPSPLDecoderType,
            inOutputModuleType
        ) -> None:
        super().__init__()

        self.EmbedDim               = inEmbeddingDim

        if isinstance(inEmbedLvlCntORList, tuple) or isinstance(inEmbedLvlCntORList, list):
            AllDims                 = [*(self.EmbedDim * i for i in inEmbedLvlCntORList)]
        else:
            AllDims                 = [*(self.EmbedDim * (2 ** i) for i in range(0, inEmbedLvlCntORList))]

        # AllDims = (1, 3, 6, 12, 24, 48, 96) 
        # list(zip(AllDims[:-1], AllDims[1:])) -> ((1, 3, 6, 12, 24, 48), (3, 6, 12, 24, 48, 96))
        InOutPairDims               = list(zip(AllDims[:-1], AllDims[1:]))

        # 1 -> input
        self.InputModule            = inInputModuleType(inInputDim, AllDims[0])

        # 2 -> downsample
        self.DSEncoderList          = nn.ModuleList([])
        for InDim, OutDim in InOutPairDims:
            self.DSEncoderList.append(nn.ModuleList([
                nn.MaxPool2d(2),
                inDNSPLEncoderType(InDim, OutDim)
            ]))

        # 3 -> Mid
        self.MidMLM                 = inMidMLMType(AllDims[-1], AllDims[-1])

        # 4 -> upsample
        self.USDecoderList          = nn.ModuleList([])
        for OutDim, InDim  in reversed(InOutPairDims):
            self.USDecoderList.append(nn.ModuleList([
                inUPSPLDecoderType(InDim * 2, OutDim),
                nn.Upsample(scale_factor=2, mode="billinear")
            ]))

        # 5 -> output
        self.OutputModule           = inOutputModuleType(AllDims[0], inOutputDim)

    def forward(self, inData):

        Stack = []

        # 1
        X = self.InputModule(inData)

        # 2
        for DownSample, Encoder in self.DSEncoderList:
            X = DownSample(X)
            X = Encoder(X)
            Stack.append(X)

        # 3
        X = self.MidMLM(X)
        
        # 4
        for Decoder, UpSample in self.USDecoderList:
            # dim=1 是除Batch后的一个维度,这个维度很可能是EmbedDim
            X = torch.cat((X, Stack.pop()), dim=1)
            X = Decoder(X)
            X = UpSample(X)

        # 5
        return self.OutputModule(X)

######################################################################################
######################################################################################
######################################################################################

class UNet2DBaseWithExtData(nn.Module):
    def __init__(
            self,
            inInputDim,
            inOutputDim,
            inEmbeddingDim,
            inEmbedLvlCntORList,
            inInputModuleType,
            inDNSPLEncoderType,   #Downsample Encoder
            inMidMLMType,         #Mid Multi Layer Module
            inUPSPLDecoderType,   #Upsample Decoder
            inOutputModuleType,
            inExtDataModuleType,
            inExtDataDim =  None
        ) -> None:
        super().__init__()

        ExtDataDim                  = inExtDataDim if inExtDataDim is not None else inEmbeddingDim

        if isinstance(inEmbedLvlCntORList, tuple) or isinstance(inEmbedLvlCntORList, list):
            AllDims                 = [*(inEmbeddingDim * i for i in inEmbedLvlCntORList)]
        else:
            AllDims                 = [*(inEmbeddingDim * (2 ** i) for i in range(0, inEmbedLvlCntORList))]

        # AllDims = (1, 3, 6, 12, 24, 48, 96) 
        # list(zip(AllDims[:-1], AllDims[1:])) -> ((1, 3, 6, 12, 24, 48), (3, 6, 12, 24, 48, 96))
        InOutPairDims               = list(zip(AllDims[:-1], AllDims[1:]))

        # 1 -> input
        self.InputModule            = inInputModuleType(inInputDim, AllDims[0])

        # 2 -> downsample
        self.DSEncoderList          = nn.ModuleList([])
        for (InDim, OutDim) in InOutPairDims:
            self.DSEncoderList.append(nn.ModuleList([
                DownsampleModule2D_PixelShuffle(InDim),
                DoubleLinearModule(ExtDataDim, InDim),
                inDNSPLEncoderType(InDim, OutDim)
            ]))

        # 3 -> Mid
        self.MidExtDataProc         = DoubleLinearModule(ExtDataDim, AllDims[-1])
        self.MidMLM                 = inMidMLMType(AllDims[-1], AllDims[-1])

        # 4 -> upsample
        self.USDecoderList          = nn.ModuleList([])
        for (OutDim, InDim) in reversed(InOutPairDims):
            self.USDecoderList.append(nn.ModuleList([
                ## 因为需要cat 所以InDim需要 * 2
                DoubleLinearModule(ExtDataDim, InDim * 2),
                inUPSPLDecoderType(InDim * 2, OutDim),
                UpsampleModule2D(OutDim)
            ]))

        # 5 -> output
        self.OutputModule           = inOutputModuleType(AllDims[0], inOutputDim)

        ### ExtData
        self.ExtDataModule          = inExtDataModuleType(ExtDataDim)

    ##########################
    def forward(self, inData, inExtData):

        ProcessedExtData            = self.ExtDataModule(inExtData)
        # ProcessedExtraData:size(Batch, EmbedDim)
        Stack = []

        # 1
        X = self.InputModule(inData)

        # 2
        for DownSample, ExtDataProc, Encoder in self.DSEncoderList:
            X = DownSample(X)
            E = ExtDataProc(ProcessedExtData)
            # E2 = rearrange(E, "b c -> b c 1 1")
            # E2 = E.unsqueeze(2).unsequeeze(3)
            X = Encoder(X, E)
            Stack.append(X)

        # 3
        E = self.MidExtDataProc(ProcessedExtData)
        # E2 = rearrange(E, "b c -> b c 1 1")
        # E2 = E.unsqueeze(2).unsequeeze(3)
        X = self.MidMLM(X, E)
        
        # 4
        for ExtDataProc, Decoder, UpSample in self.USDecoderList:
            # dim=1 是除Batch后的一个维度,这个维度很可能是EmbedDim
            X = torch.cat((X, Stack.pop()), dim=1)
            E = ExtDataProc(ProcessedExtData)
            # E2 = rearrange(E, "b c -> b c 1 1")
            # E2 = E.unsqueeze(2).unsequeeze(3)
            X = Decoder(X, E)
            X = UpSample(X)

        # 5
        return self.OutputModule(X)
