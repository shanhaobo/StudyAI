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
###################################
import os
from KongMing.Utils.OutputPath import BuildOutputPath
OutputPath = BuildOutputPath(__file__)
###########
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
    EmbeddingDim    : int    = 32
    ImageSize       : int    = 64
    ImageColorChan  : int    = 1
    SaveInterval    : int    = 13
    PrintInterval   : int    = 100

Config = TrainConfig()
Overridden = set(ApplyConfigFromKV(Config))
if not Config.bFashionMNIST:
    if "EmbeddingDim"   not in Overridden: Config.EmbeddingDim   = 128
    if "ImageColorChan" not in Overridden: Config.ImageColorChan = 3

ModelFolderByDataset    = "FashionMNIST" if Config.bFashionMNIST else "CartoonFace"
ModelRootFolderPath     = BuildOutputPath(__file__, ModelFolderByDataset)

if __name__ == "__main__" :
    if Overridden:
        print("[Config] CLI overrides:", sorted(Overridden))
    print("[HW] {}".format(FormatProfileLine(HardwareProfile)))
    GAN = DCGANModelFactory(Config.ImageColorChan, Config.EmbeddingDim, 3, inModelRootFolderPath=ModelRootFolderPath)
    Exec = Executor(GAN)

    if (Exec.ForceTrain() == False) and Exec.IsExistModel():
        GenImage = Exec.Eval(
            inImageSize=Config.ImageSize,
            inColorChanNum=Config.ImageColorChan,
            inBatchSize=15
        )

        print(GenImage.size())

        transform = transforms.Compose([
            transforms.Normalize((-1.0,), (2.0,)), #(-1, 1) -> (0, 1),
            #transforms.ToPILImage(), # turn into shape HWC, (0, 1) -> (0, 255)
        ])
        ImagetFolderPath = os.path.join(OutputPath, "images")
        os.makedirs(ImagetFolderPath, exist_ok=True)
        save_image(transform(GenImage), "{}/{}.png".format(ImagetFolderPath, datetime.now().strftime("%Y%m%d%H%M%S")), nrow=5, normalize=True)
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
