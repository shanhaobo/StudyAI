import torch
import torchvision

from datetime import datetime
from dataclasses import dataclass

from torchvision.utils import save_image

from KongMing.ModelFactory.DiffusionModel.DDPMModelFactory import DDPMModelFactory
from KongMing.Utils.Executor import Executor

from torchvision import transforms
from torch.utils.data import DataLoader
#from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

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
    # 数据集开关：True=FashionMNIST / False=CartoonFace
    bFashionMNIST   : bool   = True
    # FashionMNIST 默认 (32, 1)；CartoonFace 自动改成 (128, 3)，除非显式 --EmbeddingDim/--ImageColorChan 覆盖
    EmbeddingDim    : int    = 32
    ImageSize       : int    = 64
    ImageColorChan  : int    = 1
    # 训练超参
    LearningRate    : float  = 0.0002
    Betas           : tuple  = (0.9, 0.999)
    Timesteps       : int    = 1000
    SaveInterval    : int    = 13
    PrintInterval   : int    = 100

Config = TrainConfig()
# CLI 覆盖：python 005_DDPM.py new --LearningRate=1e-4 --bFashionMNIST=false --Betas=0.5,0.999
Overridden = set(ApplyConfigFromKV(Config))
# bFashionMNIST=False 时套上 CartoonFace 默认（用户未显式覆盖才回填）
if not Config.bFashionMNIST:
    if "EmbeddingDim"   not in Overridden: Config.EmbeddingDim   = 128
    if "ImageColorChan" not in Overridden: Config.ImageColorChan = 3

ModelFolderByDataset    = "FashionMNIST" if Config.bFashionMNIST else "CartoonFace"
ModelRootFolderPath     = BuildOutputPath(__file__, ModelFolderByDataset)

if __name__ == "__main__" :
    if Overridden:
        print("[Config] CLI overrides:", sorted(Overridden))
    print("[Run] Dataset={} | LR={} | Betas={} | Timesteps={} | EmbDim={} | ImgSize={}".format(
        ModelFolderByDataset, Config.LearningRate, Config.Betas, Config.Timesteps,
        Config.EmbeddingDim, Config.ImageSize
    ))
    print("[HW] {}".format(FormatProfileLine(HardwareProfile)))

    DDPM = DDPMModelFactory(
        inEmbeddingDim=Config.EmbeddingDim,
        inColorChanNum=Config.ImageColorChan,
        inLearningRate=Config.LearningRate,
        inBetas=Config.Betas,
        inTimesteps=Config.Timesteps,
        inModelRootFolderPath=ModelRootFolderPath
    )
    Exec = Executor(DDPM)

    if (Exec.ForceTrain() == False) and Exec.IsExistModel():
        GenImage = Exec.Eval(
            inImageSize=Config.ImageSize,
            inColorChanNum=Config.ImageColorChan,
            inBatchSize=15
        )

        # 图片落到与 checkpoint 同根的子目录，避免 FashionMNIST/CartoonFace 互相混淆
        Path = os.path.join(ModelRootFolderPath, "images")
        os.makedirs(Path, exist_ok=True)
        # save_image 自带反归一化：value_range=(-1, 1) + normalize=True 把 (-1, 1) 直接映射到 (0, 1)
        save_image(
            GenImage,
            "{}/{}.png".format(Path, datetime.now().strftime("%Y%m%d%H%M%S")),
            nrow=5,
            normalize=True,
            value_range=(-1, 1),
        )
    else:
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
            dataset = torchvision.datasets.ImageFolder(root='{}/cartoon_faces'.format(DatasetPath), transform=transform)

        dataloader = DataLoader(
            dataset,
            batch_size=HardwareProfile["BatchSize"],
            num_workers=HardwareProfile["NumWorkers"],
            pin_memory=HardwareProfile["PinMemory"],
            shuffle=True,
        )
        Exec.Train(dataloader, SaveInterval=Config.SaveInterval, PrintInterval=Config.PrintInterval)
