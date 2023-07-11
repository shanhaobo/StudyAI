import torch
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from KongMing.Models.GANs.DCGANModel import DCGANModel

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

EmbeddingDim        = 64
ImageDim            = 64
ImageColorChan      = 3

if __name__ == "__main__" :
    GAN = DCGANModel(ImageColorChan, (EmbeddingDim, 4, 4), EmbeddingDim, 4, inModelRootlFolderPath="{}/trained_models".format(OutputPath))
    Exec = Executor(GAN)

    if (Exec.ForceTrain() == False) and Exec.IsExistModel():
        GenImage = Exec.Eval(inBatchSize=15)
        
        print(GenImage.size())
        
        transform = transforms.Compose([
            transforms.Normalize((-0.5,), (2.0,)),
            transforms.Lambda(lambda t : (t + 1) * 0.5)
        ])
        ImagetFolderPath = "{}/images".format(OutputPath)
        os.makedirs(ImagetFolderPath, exist_ok=True)
        save_image(transform(GenImage), "{}/{}.png".format(ImagetFolderPath, datetime.now().strftime("%Y%m%d%H%M%S")), nrow=5, normalize=True)
    else :
        if DatasetPath is None:
            sys.exit()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t : (t * 2) - 1),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if False :
            dataset = torchvision.datasets.FashionMNIST(
                root=DatasetPath, train=True, transform=transform, download=True
            )
        else:
            dataset = datasets.ImageFolder(root='{}/cartoon_faces'.format(DatasetPath), transform=transform)
        
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        Exec.Train(dataloader, SaveInterval=13)
