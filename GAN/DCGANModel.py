import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.nn.functional as F

from Trainer.GANTrainer import GANTrainer

import os

from GANModel import GANModel

class DCGANModel(GANModel):
    
    # G(z)
    class InnerGenerator(nn.Module):
        # initializers
        def __init__(self, inDim=128, inChannel = 3):
            super(self).__init__()
            self.deconv1 = nn.ConvTranspose2d(inDim, inDim*8, 4, 1, 0)
            self.deconv1_bn = nn.BatchNorm2d(inDim*8)
            self.deconv2 = nn.ConvTranspose2d(inDim*8, inDim*4, 4, 2, 1)
            self.deconv2_bn = nn.BatchNorm2d(inDim*4)
            self.deconv3 = nn.ConvTranspose2d(inDim*4, inDim*2, 4, 2, 1)
            self.deconv3_bn = nn.BatchNorm2d(inDim*2)
            self.deconv4 = nn.ConvTranspose2d(inDim*2, inDim, 4, 2, 1)
            self.deconv4_bn = nn.BatchNorm2d(inDim)
            self.deconv5 = nn.ConvTranspose2d(inDim, inChannel, 4, 2, 1)

        # forward method
        def forward(self, input):
            #print(f"Generator:input({input.size()})")
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

    class InnerDiscriminator(nn.Module):
        # initializers
        def __init__(self, inDim=128, inChannel = 3):
            super(self).__init__()
            self.conv1 = nn.Conv2d(inChannel, inDim, 4, 2, 1)
            self.conv2 = nn.Conv2d(inDim, inDim*2, 4, 2, 1)
            self.conv2_bn = nn.BatchNorm2d(inDim*2)
            self.conv3 = nn.Conv2d(inDim*2, inDim*4, 4, 2, 1)
            self.conv3_bn = nn.BatchNorm2d(inDim*4)
            self.conv4 = nn.Conv2d(inDim*4, inDim*8, 4, 2, 1)
            self.conv4_bn = nn.BatchNorm2d(inDim*8)
            self.conv5 = nn.Conv2d(inDim*8, 1, 1, 1, 0)

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

    def __init__(self, inDim, inChannel, inGeneratorSize, inLearningRate=0.00001, inModelFolderPath=".") -> None:
        super().__init__(
            DCGANModel.InnerGenerator(inDim, inChannel),
            DCGANModel.InnerDiscriminator(inDim, inChannel),
            inGeneratorSize,
            inLearningRate,
            inModelFolderPath
        )
