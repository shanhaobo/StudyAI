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
###################################
'''
# 定义数据集
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

        images = []
        for i in range(len(train_set)):
            image, _ = train_set[i]
            images.append(image)
        self.images = images

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index]
'''

EmbeddingDim    = 64
ImageDim        = 64
ImageChannel    = 3

if __name__ == "__main__" :
    GAN = DCGANModel(EmbeddingDim, ImageChannel, (EmbeddingDim, 3, 3), inModelRootlFolderPath="{}/trained_models".format(OutputPath))
    Exec = Executor(GAN)

    if (Exec.ForceTrain() == False) and Exec.IsExistModel():
        GenImage = Exec.Eval()
        print(GenImage.size())
        
        transform = transforms.Compose([
            transforms.Normalize((-0.5,), (2.0,)),
            transforms.Lambda(lambda t : (t + 1) * 0.5)
        ])
        ImagetFolderPath = "{}/images".format(OutputPath)
        os.makedirs(ImagetFolderPath, exist_ok=True)
        save_image(transform(GenImage), "{}/{}.png".format(ImagetFolderPath, datetime.now().strftime("%Y%m%d%H%M%S")), nrow=5, normalize=True)
    else :
        #dataset = MNISTDataset()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t : (t * 2) - 1),
            transforms.Normalize((0.5,), (0.5,))
        ])
        #dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        #dataset = datasets.ImageFolder(root='D:/__DevAI__/Datasets/cartoon_faces', transform=transform)
        dataset = datasets.ImageFolder(root='D:/AI/Datasets/cartoon_faces', transform=transform)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        Exec.Train(dataloader, SaveModelInterval=10)
