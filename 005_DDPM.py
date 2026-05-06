import torch
import torchvision

from datetime import datetime

from torchvision.utils import save_image

from KongMing.ModelFactory.DiffusionModel.DDPMModelFactory import DDPMModelFactory
from KongMing.Utils.Executor import Executor

from torchvision import transforms
from torch.utils.data import DataLoader
#from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

###################################
import os
OutputPath = "output/{}".format(os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(OutputPath, exist_ok=True)
###########
from KongMing.Utils.DatasetPath import ResolveDatasetPath
DatasetPath = ResolveDatasetPath()

###################################

torch.set_printoptions(precision=10, sci_mode=False)

###################################

bFashionMNIST       = True
if bFashionMNIST:
    ModelFolderByDataset    = "FashionMNIST"
    EmbeddingDim            = 32
    ImageSize               = 64
    ImageColorChan          = 1
else:
    ModelFolderByDataset    = "CartoonFace"
    EmbeddingDim            = 128
    ImageSize               = 64
    ImageColorChan          = 3

ModelRootFolderPath     = "{}/{}".format(OutputPath, ModelFolderByDataset)

LearningRate            = 0.0002
Timesteps               = 1000
# Betas 在 KongMing/Trainer/DDPMTrainer.py:_CreateOptimizer 里硬编码，这里只用于打印
BetasForLog             = (0.9, 0.999)

if __name__ == "__main__" :
    print("[Run] Dataset={} | LR={} | Betas={} | Timesteps={} | EmbDim={} | ImgSize={}".format(
        ModelFolderByDataset, LearningRate, BetasForLog, Timesteps, EmbeddingDim, ImageSize
    ))

    DDPM = DDPMModelFactory(
        inEmbeddingDim=EmbeddingDim,
        inColorChanNum= ImageColorChan,
        inLearningRate=LearningRate,
        inTimesteps=Timesteps,
        inModelRootFolderPath=ModelRootFolderPath
    )
    Exec = Executor(DDPM)

    if (Exec.ForceTrain() == False) and Exec.IsExistModel():
        GenImage = Exec.Eval(
            inImageSize=ImageSize,
            inColorChanNum=ImageColorChan,
            inBatchSize=15
        )
        print(GenImage.size())
        
        reverse_transform = transforms.Compose([
            transforms.Normalize((-1.0,), (2.0,)), #(-1, 1) -> (0, 1),
            #transforms.ToPILImage(), # turn into shape HWC, (0, 1) -> (0, 255)
        ])
        Path = "{}/images".format(OutputPath)
        os.makedirs(Path, exist_ok=True)
        save_image(reverse_transform(GenImage), "{}/{}.png".format(Path, datetime.now().strftime("%Y%m%d%H%M%S")), nrow=5, normalize=True)
    else:
        transform = transforms.Compose([
            transforms.Resize(ImageSize),
            transforms.ToTensor(), # HWC -> CHW, (0, 255) -> (0, 1), 
            transforms.Normalize((0.5,), (0.5,))  # (0, 1) -> (-1, 1),
        ])
        if bFashionMNIST :
            dataset = torchvision.datasets.FashionMNIST(
                root=DatasetPath, train=True, transform=transform, download=True
            )
        else:
            dataset = torchvision.datasets.ImageFolder(root='{}/cartoon_faces'.format(DatasetPath), transform=transform)

        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        Exec.Train(dataloader, SaveInterval=13)
