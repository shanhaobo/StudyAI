import torch
import torchvision
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from KongMing.ModelFactory.GANs.DCGANModelFactory import DCGANModelFactory

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

bFashionMNIST       = True
if bFashionMNIST:
    ModelFolderByDataset    = "FashionMNIST"
    EmbeddingDim            = 32
    ImageSize               = 64
    ImageColorChan          = 1
else:
    ModelFolderByDataset    = "CartoonnFace"
    EmbeddingDim            = 128
    ImageSize               = 64
    ImageColorChan          = 3

ModelRootFolderPath     = "{}/{}".format(OutputPath, ModelFolderByDataset)

if __name__ == "__main__" :
    GAN = DCGANModelFactory(ImageColorChan, EmbeddingDim, 3, inModelRootFolderPath=ModelRootFolderPath)
    Exec = Executor(GAN)

    if (Exec.ForceTrain() == False) and Exec.IsExistModel():
        GenImage = Exec.Eval(
            inImageSize=ImageSize,
            inColorChanNum=ImageColorChan,
            inBatchSize=15
        )
        
        print(GenImage.size())
        
        transform = transforms.Compose([
            transforms.Normalize((-1.0,), (2.0,)), #(-1, 1) -> (0, 1),
            #transforms.ToPILImage(), # turn into shape HWC, (0, 1) -> (0, 255)
        ])
        ImagetFolderPath = "{}/images".format(OutputPath)
        os.makedirs(ImagetFolderPath, exist_ok=True)
        save_image(transform(GenImage), "{}/{}.png".format(ImagetFolderPath, datetime.now().strftime("%Y%m%d%H%M%S")), nrow=5, normalize=True)
    else :
        if DatasetPath is None:
            sys.exit()

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
            dataset = datasets.ImageFolder(root='{}/cartoon_faces'.format(DatasetPath), transform=transform)
        
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        Exec.Train(dataloader, SaveInterval=13)
