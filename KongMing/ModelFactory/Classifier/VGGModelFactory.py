from KongMing.Archiver.SingleNNArchiver import SingleNNArchiver
from KongMing.ModelFactory.BaseModelFactory import BaseModelFactory
from KongMing.Trainer.VGGTrainer import VGGTrainer

from KongMing.Models.BaseNNModel import BaseNNModel

from KongMing.Utils.CaseInsensitiveContainer import CaseInsensitiveList, CaseInsensitiveDict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import VGG16_Weights

class VGG16(BaseNNModel):
    def __init__(self, inNumClasses=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, inNumClasses),
        )

    def forward(self, inX):
        inX = self.features(inX)
        inX = self.avgpool(inX)
        inX = torch.flatten(inX, 1)
        inX = self.classifier(inX)
        return inX

class VGGModelFactory(BaseModelFactory) :
    def __init__(
            self,
            inNumClasses,
            inLearningRate,
            inModelRootFolderPath
        ):
        self.VGG = VGG16(inNumClasses)

        Trainer = VGGTrainer(self.VGG, inLearningRate, inModelRootFolderPath)
        Archiver = SingleNNArchiver(self.VGG, inModelRootFolderPath)

        super().__init__(Trainer, Archiver)

        print("Sum of Params:{:,} ".format(self._SumParameters(self.VGG)))

    def NewTrain(self, inDataLoader, inEpochIterCount : int, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None) -> None:
        PreTrained = inArgs.get("LoadPretrained")
        if PreTrained is not None:
            PreTrainedModel = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.VGG.features.load_state_dict(PreTrainedModel.features.state_dict())

        super().NewTrain(inDataLoader=inDataLoader, inEpochIterCount=inEpochIterCount, inArgs=inArgs, inKVArgs=inKVArgs)

    def Eval(self, inEpoch, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None):
        if (super().Eval(inEpoch, inArgs, inKVArgs) == False) :
            return

        TestDataLoader = inKVArgs.get("inDataLoader")
        if (TestDataLoader is None) :
            return

        self.VGG.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in TestDataLoader:
                images, labels = data[0].to(self.Device), data[1].to(self.Device)
                outputs = self.VGG(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
