import torch
import torch.nn as nn

from .GANModelFactory import GANModelFactory

from KongMing.Models.BaseNNModel import BaseNNModel

class DCGANModelFactory(GANModelFactory):
    # G(z)
    class InnerGenerator(BaseNNModel):
        # initializers
        def __init__(self, inColorChan, inAllEmbeddingDims):
            super().__init__()
            # [In, Out] -> [Out, In]
            # [[0, Len - 1], [1, Len]] -> [[1, Len], [0, Len - 1]] -> reversed[[Len, 1], [Len - 1, 0]]
            # [(2048, 1024), (1024, 512), (512, 256), (256, 128), (128, 64), (64, 32), (32, 0)]
            InOutPairDims = reversed(list(zip(inAllEmbeddingDims[1:], inAllEmbeddingDims[:-1])))

            self.InputModule = nn.Sequential(
                nn.ConvTranspose2d(inAllEmbeddingDims[0], inAllEmbeddingDims[-1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(inAllEmbeddingDims[-1]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2, stride=2),
            )
            self.ModuleList = nn.ModuleList([])
            for InDim, OutDim in InOutPairDims:
                self.ModuleList.append(nn.Sequential(
                    nn.ConvTranspose2d(InDim, OutDim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(OutDim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.AvgPool2d(2, stride=2),
                ))
            self.FinalModule = nn.Sequential(
                nn.ConvTranspose2d(inAllEmbeddingDims[0], inColorChan, kernel_size=4, stride=2, padding=1),
                nn.AvgPool2d(2, stride=2),
                nn.Tanh()
            )

        # forward method
        def forward(self, inData):
            X = self.InputModule(inData)
            
            for Module in self.ModuleList:
                X = Module(X)

            return self.FinalModule(X)

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
            X = self.InputModule(inData)

            for Module in self.ModuleList:
                X = Module(X)

            return self.FinalModule(X)

    def __init__(self, inColorChan, inEmbeddingDim, inEmbedLvlCntORList, inLearningRate=0.00001, inModelRootFolderPath=".") -> None:

        if isinstance(inEmbedLvlCntORList, tuple) or isinstance(inEmbedLvlCntORList, list):
            AllEmbeddingDims = [*(inEmbeddingDim * i for i in inEmbedLvlCntORList)]
        else:
            AllEmbeddingDims = [*(inEmbeddingDim * (2 ** i) for i in range(0, inEmbedLvlCntORList))]

        self.Generator = DCGANModelFactory.InnerGenerator(
            inColorChan=inColorChan,
            inAllEmbeddingDims=AllEmbeddingDims
        )

        self.Discriminator = DCGANModelFactory.InnerDiscriminator(
            inColorChan=inColorChan,
            inAllEmbeddingDims=AllEmbeddingDims
        )

        super().__init__(
            self.Generator,
            self.Discriminator,
            inEmbeddingDim,
            inWTrainer=False,
            inLearningRate = inLearningRate,
            inModelRootFolderPath=inModelRootFolderPath
        )
