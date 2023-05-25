import torch
from torch import nn
from torch.utils.data import DataLoader

import torch.nn.functional as F

import torchvision.datasets as datasets
from torchvision.transforms import transforms

from GAN.GANModel import GANModel

# G(z)
class Generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(d, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # forward method
    def forward(self, input):
        #print(f"Generator:input({input.size()})")
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        #print(f"Generator:deconv1({x.size()})")
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        #print(f"Generator:deconv2({x.size()})")
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        #print(f"Generator:deconv3({x.size()})")
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        #print(f"Generator:deconv4({x.size()})")
        x = torch.tanh(self.deconv5(x))
        #print(f"Generator:deconv5({x.size()})")

        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 1, 1, 0)

    # forward method
    def forward(self, input):
        #print(f"Discriminator:input({input.size()})")
        x = F.leaky_relu(self.conv1(input), 0.2)
        #print(f"Discriminator:conv1({x.size()})")
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        #print(f"Discriminator:conv2({x.size()})")
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        #print(f"Discriminator:conv3({x.size()})")
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        #print(f"Discriminator:conv4({x.size()})")
        x = torch.sigmoid(self.conv5(x))
        #print(f"Discriminator:conv5({x.size()})")

        return x

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

if __name__ == "__main__" :
    GAN = GANModel(Generator(128), Discriminator(128), (128, 1, 1), inModelPath="./models/GAN")
    if GAN.IsExistModels() :
        GAN.Gen()
    else :
        dataset = MNISTDataset()
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
        GAN.Train(10, dataloader, inSaveModelInterval=1)
