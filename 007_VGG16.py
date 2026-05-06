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
from KongMing.Utils.DatasetPath import ResolveDatasetPath
from KongMing.Utils.HardwareProfile import DetectHardwareProfile, FormatProfileLine
DatasetPath = ResolveDatasetPath()
# VGG16 + 224x224 显存占用约为 DDPM/DCGAN 基线的 4 倍
HardwareProfile = DetectHardwareProfile(inMemoryFactor=4.0)

###################################

torch.set_printoptions(precision=10, sci_mode=False)

###################################

ImageSizeW          = 224
ImageSizeH          = 224
NumClasses          = 10

if __name__ == "__main__" :
    print("[HW] {}".format(FormatProfileLine(HardwareProfile)))

    VGG = VGGModelFactory(NumClasses, inLearningRate=0.0001, inModelRootFolderPath="{}/CIFAR10".format(OutputPath))
    Exec = Executor(VGG)

    # 当前是Eval 还是 Train
    DoEval =  (Exec.ForceTrain() == False) and Exec.IsExistModel()

    # 加载相应数据
    transform = transforms.Compose([
        transforms.Resize((ImageSizeW, ImageSizeH)),
        transforms.ToTensor(), # HWC -> CHW, (0, 255) -> (0, 1), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # (0, 1) -> (-1, 1),
    ])

    dataset = torchvision.datasets.CIFAR10(root=DatasetPath, train=(DoEval == False), download=True, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=HardwareProfile["BatchSize"],
        num_workers=HardwareProfile["NumWorkers"],
        pin_memory=HardwareProfile["PinMemory"],
        shuffle=True,
    )

    # 开始Eval 或者 Train
    if DoEval:
        Exec.Eval(inDataLoader=dataloader)
    else :
        Exec.Train(dataloader, SaveInterval=1, PrintInterval=100)
