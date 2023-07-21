import torch
import torchvision
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from KongMing.ModelFactory.Classifier.VGGModelFactory import VGGModelFactory

from datetime import datetime

from KongMing.Utils.Executor import Executor
###################################
import os
OutputPath = "output/{}".format(os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(OutputPath, exist_ok=True)
###########
DatasetPath = None
if os.path.exists("D:/AI/") :
    DatasetPath = "D:/AI/"
elif os.path.exists("D:/__DevAI__/") :
    DatasetPath = "D:/__DevAI__/"
DatasetPath = os.path.join(DatasetPath, "Datasets")

import sys

###################################

torch.set_printoptions(precision=10, sci_mode=False)

###################################

EmbeddingDim        = 256
ImageSizeW          = 224
ImageSizeH          = 224
ImageColorChan      = 1

if __name__ == "__main__" :
    VGG = VGGModelFactory(inLearningRate=0.00001, inModelRootFolderPath="{}/trained_models".format(OutputPath))
    Exec = Executor(VGG)

    transform = transforms.Compose([
        transforms.Resize((ImageSizeW, ImageSizeH)),
        transforms.ToTensor(), # HWC -> CHW, (0, 255) -> (0, 1), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # (0, 1) -> (-1, 1),
    ])

    if DatasetPath is None:
        sys.exit()

    if (Exec.ForceTrain() == False) and Exec.IsExistModel():
        testset = torchvision.datasets.CIFAR10(root=DatasetPath, train=False,
                                            download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
        Exec.Eval(inDataLoader=testloader)
    else :
        trainset = torchvision.datasets.CIFAR10(root=DatasetPath, train=True, download=True, transform=transform)
        
        dataloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        Exec.Train(dataloader, SaveInterval=13)
