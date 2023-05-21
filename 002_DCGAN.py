import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

# 训练参数
batch_size = 128
lr = 0.0002
epoch = 100

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 改变图像大小，例如将所有图像改为 224x224
    transforms.ToTensor(),  # 将图像转换为 PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化，这些均值和方差是预训练的 ImageNet 模型常用的参数
])
# 数据加载
'''
dataset = datasets.MNIST(root='~/data/', 
                         transform=transforms.ToTensor(),
                         download=True)
'''
dataset = datasets.ImageFolder(root='D:/__DevAI__/DataSets/cartoon_faces', transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义生成器和判别器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# 初始化模型
gen = Generator()
dis = Discriminator()
gen_opt = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
dis_opt = optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999))

# 训练
for epoch in range(epoch):
    for i, (x, _) in enumerate(loader):
        # 判别器训练
        dis.zero_grad()
        dis_real = dis(x).mean()
        dis_real.backward()

        noise = torch.randn(batch_size, 100, 1, 1)
        gen_fake = gen(noise)
        dis_fake = dis(gen_fake.detach()).mean()
        dis_fake.backward()
        dis_opt.step()
 
        # 生成器训练
        gen.zero_grad()
        dis_fake = dis(gen_fake)
        gen_loss = -dis_fake.mean()
        gen_loss.backward()
        gen_opt.step()
