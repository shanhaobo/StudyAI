import torch
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from Models.GANs.DCGANModel import DCGANModel

from datetime import datetime

from Utils.Executor import Executor

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

if __name__ == "__main__" :
    GAN = DCGANModel(128, 3, (128, 1, 1), inModeRootlFolderPath="./trained_models/CFGAN")
    Exec = Executor(GAN)

    if Exec.IsExistModel() and Exec.ReadyTrain() == False:
        GenImage = Exec.Eval()
        print(GenImage.size())
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        save_image(GenImage, "images/{}.png".format(timestamp), nrow=5, normalize=True)
    else :
        #dataset = MNISTDataset()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        #dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        #dataset = datasets.ImageFolder(root='D:/__DevAI__/Datasets/cartoon_faces', transform=transform)
        dataset = datasets.ImageFolder(root='D:/AI/Datasets/cartoon_faces', transform=transform)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        Exec.Train(dataloader, SaveModelInterval=1)
