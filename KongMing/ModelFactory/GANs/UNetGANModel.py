import torch
import torch.nn as nn

from .GANModelFactory import GANModelFactory
from KongMing.Models.UNets.UNet2DBase import UNet2DBase

from KongMing.Models.BaseNNModel import BaseNNModel

#########################################################################

class UNet2D_GAN_InitConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim) -> None:
        super().__init__()

        self.Blocks = nn.Sequential(
            # kernel_size=5, stride=1, padding=2 保持大小不变
            nn.Conv2d(inInputDim, inOutputDim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(inOutputDim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, inData):
        return self.Blocks(inData)
    
class UNet2D_GAN_SampleConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim) -> None:
        super().__init__()

        self.Blocks = nn.Sequential(
                # kernel_size=3, stride=1, padding=1 保持大小不变
                nn.Conv2d(inInputDim, inOutputDim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(inOutputDim),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, inData):
        return self.Blocks(inData)

class UNet2D_GAN_FinalConv(nn.Module):
    def __init__(self, inInputDim, inOutputDim) -> None:
        super().__init__()

        self.Blocks = nn.Sequential(
            # kernel_size=5, stride=1, padding=2 保持大小不变
            nn.Conv2d(inInputDim, inOutputDim, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(inOutputDim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh()
        )

    def forward(self, inData):
        return self.Blocks(inData)
    
#########################################################################

class UNet2D_GAN(UNet2DBase) :
    def __init__(self, inDim, inColorChanNum, inEmbeddingDim, inEmbedLvlCntORList) -> None:
        super().__init__(
            inDim,
            inColorChanNum,
            inEmbeddingDim,
            inEmbedLvlCntORList,
            UNet2D_GAN_InitConv,
            UNet2D_GAN_SampleConv,
            UNet2D_GAN_InitConv,
            UNet2D_GAN_SampleConv,
            UNet2D_GAN_FinalConv
        )

#########################################################################

class UNetGANModelFactory(GANModelFactory):
    
    class InnerDiscriminator(BaseNNModel):
        # initializers
        def __init__(self, inColorChan, inAllEmbeddingDims):
            super().__init__()

            InOutPairDims = list(zip(inAllEmbeddingDims[:-1], inAllEmbeddingDims[1:]))

            self.InputModule = nn.Sequential(
                # kernel_size=4, stride=2, padding=1 变为输入大小的二分之一
                nn.Conv2d(inColorChan, inAllEmbeddingDims[0], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(inAllEmbeddingDims[0]),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.ModuleList = nn.ModuleList([])
            for InDim, OutDim in InOutPairDims:
                self.ModuleList.append(nn.Sequential(
                    # kernel_size=4, stride=2, padding=1 变为输入大小的二分之一
                    nn.Conv2d(InDim, OutDim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(OutDim),
                    nn.LeakyReLU(0.2, inplace=True)
                ))
            self.FinalModule = nn.Sequential(
                # kernel_size=3, stride=1, padding=1 保持输入大小变
                nn.Conv2d(inAllEmbeddingDims[-1], 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )

        # forward method
        def forward(self, inData):
            x = self.InputModule(inData)

            for Module in self.ModuleList:
                x = Module(x)

            return self.FinalModule(x)

    def __init__(self, inColorChan, inEmbeddingDim, inEmbedLvlCntORList, inLearningRate=0.00001, inModelRootFolderPath=".") -> None:
        
        if isinstance(inEmbedLvlCntORList, tuple) or isinstance(inEmbedLvlCntORList, list):
            AllEmbeddingDims = [*(inEmbeddingDim * i for i in inEmbedLvlCntORList)]
        else:
            AllEmbeddingDims = [*(inEmbeddingDim * (2 ** i) for i in range(0, inEmbedLvlCntORList))]

        self.Generator = UNet2D_GAN(
            inEmbeddingDim,
            inColorChanNum=inColorChan,
            inEmbeddingDim=inEmbeddingDim,
            inEmbedLvlCntORList=inEmbedLvlCntORList
        )

        self.Generator.ApplyEMA(0.999)

        self.Discriminator = UNetGANModelFactory.InnerDiscriminator(
            inColorChan=inColorChan,
            inAllEmbeddingDims=AllEmbeddingDims
        )

        super().__init__(
            self.Generator,
            self.Discriminator,
            inEmbeddingDim,
            inWTrainer=True,
            inLearningRate = inLearningRate,
            inModelRootFolderPath=inModelRootFolderPath
        )

#########################################################################
