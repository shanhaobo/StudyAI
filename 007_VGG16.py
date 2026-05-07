import torch
import torchvision
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from KongMing.ModelFactory.Classifier.VGGModelFactory import VGGModelFactory

from datetime import datetime
from dataclasses import dataclass

from KongMing.Utils.Executor import Executor
###################################
import os
from KongMing.Utils.OutputPath import BuildOutputPath
OutputPath = BuildOutputPath(__file__)
###########
from KongMing.Utils.DatasetPath import ResolveDatasetPath
from KongMing.Utils.HardwareProfile import DetectHardwareProfile, FormatProfileLine
from KongMing.Utils.ConfigUtils import ApplyConfigFromKV
DatasetPath = ResolveDatasetPath()
# VGG16 + 224x224 显存占用约为 DDPM/DCGAN 基线的 4 倍
HardwareProfile = DetectHardwareProfile(inMemoryFactor=4.0)

###################################

torch.set_printoptions(precision=10, sci_mode=False)

###################################

@dataclass
class TrainConfig:
    ImageSizeW      : int    = 224
    ImageSizeH      : int    = 224
    NumClasses      : int    = 10
    LearningRate    : float  = 0.0001
    SaveInterval    : int    = 1
    PrintInterval   : int    = 100

Config = TrainConfig()
Overridden = set(ApplyConfigFromKV(Config))

if __name__ == "__main__" :
    if Overridden:
        print("[Config] CLI overrides:", sorted(Overridden))
    print("[HW] {}".format(FormatProfileLine(HardwareProfile)))

    VGG = VGGModelFactory(Config.NumClasses, inLearningRate=Config.LearningRate, inModelRootFolderPath=BuildOutputPath(__file__, "CIFAR10"))
    Exec = Executor(VGG)

    # 当前是Eval 还是 Train
    DoEval =  (Exec.ForceTrain() == False) and Exec.IsExistModel()

    # 加载相应数据
    transform = transforms.Compose([
        transforms.Resize((Config.ImageSizeW, Config.ImageSizeH)),
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
        Exec.Train(dataloader, SaveInterval=Config.SaveInterval, PrintInterval=Config.PrintInterval)
