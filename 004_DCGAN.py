import os
import torch
import torchvision
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from KongMing.ModelFactory.GANs.DCGANModelFactory import DCGANModelFactory

from datetime import datetime
from dataclasses import dataclass

from KongMing.Utils.Executor import Executor
from KongMing.Utils.OutputPath import BuildOutputPath
from KongMing.Utils.DatasetPath import ResolveDatasetPath
from KongMing.Utils.HardwareProfile import DetectHardwareProfile, FormatProfileLine
from KongMing.Utils.ConfigUtils import ApplyConfigFromKV

DatasetPath = ResolveDatasetPath()
HardwareProfile = DetectHardwareProfile()

###################################

torch.set_printoptions(precision=10, sci_mode=False)

###################################

@dataclass
class TrainConfig:
    bFashionMNIST   : bool   = True
    EmbeddingDim    : int    = 32      # FashionMNIST 32 / CartoonFace 128（按数据集自动覆盖默认值）
    ImageSize       : int    = 64
    ImageColorChan  : int    = 1       # FashionMNIST 1 / CartoonFace 3（同上）
    Depth           : int    = 3       # DCGAN 卷积栈层数 = embedding 倍增层数
    LearningRate    : float  = 0.00001
    SaveInterval    : int    = 13
    PrintInterval   : int    = 100

Config = TrainConfig()
Overridden = set(ApplyConfigFromKV(Config))

# 数据集相关的默认值——只在用户没用 --Key= 显式覆盖时生效。
# True / False 两边都列全，避免之前那种"只处理 not bFashionMNIST"的不对称写法。
DatasetDefaults = {
    True  : {"EmbeddingDim" :  32, "ImageColorChan" : 1},  # FashionMNIST
    False : {"EmbeddingDim" : 128, "ImageColorChan" : 3},  # CartoonFace
}
for Key, Value in DatasetDefaults[Config.bFashionMNIST].items():
    if Key not in Overridden:
        setattr(Config, Key, Value)

ModelFolderByDataset    = "FashionMNIST" if Config.bFashionMNIST else "CartoonFace"
ModelRootFolderPath     = BuildOutputPath(__file__, ModelFolderByDataset)

if __name__ == "__main__" :
    if Overridden:
        print("[Config] CLI overrides:", sorted(Overridden))
    print("[HW] {}".format(FormatProfileLine(HardwareProfile)))
    GAN = DCGANModelFactory(
        Config.ImageColorChan,
        Config.EmbeddingDim,
        Config.Depth,
        inLearningRate=Config.LearningRate,
        inModelRootFolderPath=ModelRootFolderPath
    )
    Exec = Executor(GAN)

    if (Exec.ForceTrain() == False) and Exec.IsExistModel():
        GenImage = Exec.Eval(
            inImageSize=Config.ImageSize,
            inColorChanNum=Config.ImageColorChan,
            inBatchSize=15
        )

        # 图片落到与 checkpoint 同根的子目录，避免 FashionMNIST/CartoonFace 互相混淆
        ImagetFolderPath = os.path.join(ModelRootFolderPath, "images")
        os.makedirs(ImagetFolderPath, exist_ok=True)
        # save_image 自带反归一化：value_range=(-1, 1) + normalize=True 把 (-1, 1) 直接映射到 (0, 1)
        save_image(
            GenImage,
            "{}/{}.png".format(ImagetFolderPath, datetime.now().strftime("%Y%m%d%H%M%S")),
            nrow=5,
            normalize=True,
            value_range=(-1, 1),
        )
    else :
        transform = transforms.Compose([
            transforms.Resize(Config.ImageSize),
            transforms.ToTensor(), # HWC -> CHW, (0, 255) -> (0, 1),
            transforms.Normalize((0.5,), (0.5,))  # (0, 1) -> (-1, 1),
        ])
        if Config.bFashionMNIST :
            dataset = torchvision.datasets.FashionMNIST(
                root=DatasetPath, train=True, transform=transform, download=True
            )
        else:
            dataset = datasets.ImageFolder(root='{}/cartoon_faces'.format(DatasetPath), transform=transform)

        dataloader = DataLoader(
            dataset,
            batch_size=HardwareProfile["BatchSize"],
            num_workers=HardwareProfile["NumWorkers"],
            pin_memory=HardwareProfile["PinMemory"],
            shuffle=True,
        )
        Exec.Train(dataloader, SaveInterval=Config.SaveInterval, PrintInterval=Config.PrintInterval)
