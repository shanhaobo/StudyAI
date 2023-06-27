import torch
import torch.nn as nn

from einops import rearrange

from .UtilsModules import DoubleLinearModule

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
            DNSPLEncoderType,
            MidMLMType,
            UPSPLDecoderType,
            OutputModuleType
        ) -> None:
        super().__init__()

        self.EmbedDim               = inEmbedDims

        if isinstance(inEmbedLvlCntORList, tuple) or isinstance(inEmbedLvlCntORList, list):
            AllDims                 = [*(self.EmbedDim * i for i in inEmbedLvlCntORList)]
        else:
            AllDims                 = [*(self.EmbedDim * (2 ** i) for i in range(0, inEmbedLvlCntORList + 1))]

        # AllDims = (1, 3, 6, 12, 24, 48, 96) 
        # list(zip(AllDims[:-1], AllDims[1:])) -> ((1, 3, 6, 12, 24, 48), (3, 6, 12, 24, 48, 96))
        InOutPairDims               = list(zip(AllDims[:-1], AllDims[1:]))

        # 1 -> input
        self.InputModule            = InputModuleType(inInputDim, AllDims[0])

        # 2 -> downsample
        self.DownSample             = nn.MaxPool2d(2)
        self.DSEncoderList          = nn.ModuleList([])
        for inDim, outDim in InOutPairDims:
            self.DSEncoderList.append(nn.ModuleList([
                DNSPLEncoderType(inDim, outDim)
            ]))

        # 3 -> Mid
        self.MidMLM                 = MidMLMType(AllDims[-1], AllDims[-1])

        # 4 -> upsample
        self.USDecoderList          = nn.ModuleList([])
        for inDim, outDim in reversed(InOutPairDims):
            self.USDecoderList.append(nn.ModuleList([
                UPSPLDecoderType(outDim * 2, inDim)
            ]))
        self.UpSample               = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 5 -> output
        self.OutputModule           = OutputModuleType(AllDims[0], inOutputDim)

    def forward(self, inData):

        Stack = []

        # 1
        X = self.InputModule(inData)

        # 2
        for Encoder in self.DSEncoderList:
            X = self.DownSample(X)
            X = Encoder(X)
            Stack.append(X)

        # 3
        X = self.MidMLM(X)
        
        # 4
        for Decoder in self.USDecoderList:
            # dim=1 是除Batch后的一个维度,这个维度很可能是EmbedDim
            X = torch.cat((X, Stack.pop()), dim=1)
            X = Decoder(X)
            X = self.UpSample(X)

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
            inEmbedDim,
            inEmbedLvlCntORList,
            InputModuleType,
            DNSPLEncoderType,   #Downsample Encoder
            MidMLMType,         #Mid Multi Layer Module
            UPSPLDecoderType,   #Upsample Decoder
            OutputModuleType,
            ExtDataModuleType,
            inExtEmbedDim =  None
        ) -> None:
        super().__init__()

        self.EmbedDim               = inEmbedDim
        self.ExtEmbedDim            = inEmbedDim if inExtEmbedDim is None else inExtEmbedDim

        if isinstance(inEmbedLvlCntORList, tuple) or isinstance(inEmbedLvlCntORList, list):
            AllDims                 = [*(self.EmbedDim * i for i in inEmbedLvlCntORList)]
        else:
            AllDims                 = [*(self.EmbedDim * (2 ** i) for i in range(0, inEmbedLvlCntORList + 1))]

        # AllDims = (1, 3, 6, 12, 24, 48, 96) 
        # list(zip(AllDims[:-1], AllDims[1:])) -> ((1, 3, 6, 12, 24, 48), (3, 6, 12, 24, 48, 96))
        InOutPairDims               = list(zip(AllDims[:-1], AllDims[1:]))

        # 1 -> input
        self.InputModule            = InputModuleType(inInputDim, AllDims[0])

        # 2 -> downsample
        self.DownSample             = nn.MaxPool2d(2)
        self.DSEncoderList          = nn.ModuleList([])
        for (InDim, OutDim) in InOutPairDims:
            self.DSEncoderList.append(nn.ModuleList([
                DoubleLinearModule(self.ExtEmbedDim, InDim),
                DNSPLEncoderType(InDim, OutDim)
            ]))

        # 3 -> Mid
        self.MidExtDataProc         = DoubleLinearModule(self.ExtEmbedDim, AllDims[-1])
        self.MidMLM                 = MidMLMType(AllDims[-1], AllDims[-1])

        # 4 -> upsample
        self.USDecoderList          = nn.ModuleList([])
        for (OutDim, InDim) in reversed(InOutPairDims):
            self.USDecoderList.append(nn.ModuleList([
                ## 因为需要cat 所以InDim需要 * 2
                DoubleLinearModule(self.ExtEmbedDim, InDim * 2),
                UPSPLDecoderType(InDim * 2, OutDim)
            ]))
        self.UpSample               = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 5 -> output
        self.OutputModule           = OutputModuleType(AllDims[0], inOutputDim)

        ### ExtData
        self.ExtDataModule          = ExtDataModuleType(self.ExtEmbedDim)

    ##########################
    def forward(self, inData, inExtData):

        ProcessedExtData            = self.ExtDataModule(inExtData)
        # ProcessedExtraData:size(Batch, EmbedDim)
        Stack = []

        # 1
        X = self.InputModule(inData)

        # 2
        for ExtDataProc, Encoder in self.DSEncoderList:
            X = self.DownSample(X)
            E = ExtDataProc(ProcessedExtData)
            E = rearrange(E, "b c -> b c 1 1")
            # E = E.unsqueeze(2).unsequeeze(3)
            X = self.HandleData(X, E, Encoder)
            Stack.append(X)

        # 3
        E = self.MidExtDataProc(ProcessedExtData)
        E = rearrange(E, "b c -> b c 1 1")
        # E = E.unsqueeze(2).unsequeeze(3)
        X = self.HandleData(X, E, self.MidMLM)
        
        # 4
        for ExtDataProc, Decoder in self.USDecoderList:
            # dim=1 是除Batch后的一个维度,这个维度很可能是EmbedDim
            X = torch.cat((X, Stack.pop()), dim=1)
            E = ExtDataProc(ProcessedExtData)
            E = rearrange(E, "b c -> b c 1 1")
            # E = E.unsqueeze(2).unsequeeze(3)
            X = self.HandleData(X, E, Decoder)
            X = self.UpSample(X)

        # 5
        return self.OutputModule(X)
    
    ##########################
    def HandleData(self, inData, inExtData, inFunc):
        return inFunc(inData, inExtData)

######################################################################################
######################################################################################
######################################################################################

class UNet2DBasePLUSExtData(UNet2DBaseWithExtData):
    def __init__(
            self,
            inInputDim,
            inOutputDim,
            inEmbedDims,
            inEmbedLvlCntORList,
            InputModuleType,
            DNSPLEncoderType,   #Downsample Encoder
            MidMLMType,         #Mid Multi Layer Module
            UPSPLDecoderType,   #Upsample Decoder
            OutputModuleType,
            ExtDataModuleType
        ) -> None:
        super().__init__(
            inInputDim,
            inOutputDim,
            inEmbedDims,
            inEmbedLvlCntORList,
            InputModuleType,
            DNSPLEncoderType,   #Downsample Encoder
            MidMLMType,         #Mid Multi Layer Module
            UPSPLDecoderType,   #Upsample Decoder
            OutputModuleType,
            ExtDataModuleType
        )

    ##########################
    def HandleData(self, inData, inExtData, inFunc):
        return inFunc(inData + inExtData)
