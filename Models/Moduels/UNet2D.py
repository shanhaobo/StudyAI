import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNetBase import UNetBase

# UNet的一大层，包含了两层小的卷积
class DoubleConv(nn.Module):
    def __init__(self, inInputChannels, inOutputChannels):
        super(DoubleConv, self).__init__()

        """
        self.Blocks = nn.Sequential(
            nn.Conv2d(inInputChannels, inOutputChannels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inOutputChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inOutputChannels, inOutputChannels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inOutputChannels),
            nn.ReLU(inplace=True)
        )
        """
        print("{},{}".format(inInputChannels, inOutputChannels))
        self.Blocks = nn.Conv2d(inInputChannels, inOutputChannels, kernel_size=3, padding=1, bias=False)

    def forward(self, inData):
        print(inData.size())
        return self.Blocks(inData)

# 定义输入进来的第一层
class InputConv(nn.Module):
    def __init__(self, inInputChannels, inOutputChannels):
        super(InputConv, self).__init__()
        self.Blocks = DoubleConv(inInputChannels, inOutputChannels)

    def forward(self, inData):
        return self.Blocks(inData)

# 定义encoder中的向下传播，包括一个maxpool和一大层    
class DownSample(nn.Module):
    def __init__(self, inInputChannels, inOutputChannels):
        super(DownSample, self).__init__()
        self.Blocks = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(inInputChannels, inOutputChannels)
        )

    def forward(self, inData):
        return self.Blocks(inData)

# 定义decoder中的向上传播
class UpSample(nn.Module):
    def __init__(self, inInputChannels, inOutputChannels, inBilinear=True):
        super(UpSample, self).__init__()
        # 定义了self.up的方法
        if inBilinear:
            self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.Up = nn.ConvTranspose2d(inInputChannels // 2, inInputChannels // 2, 2, stride=2) # // 除以的结果向下取整

        self.Output = DoubleConv(inInputChannels, inOutputChannels)

    def forward(self, inData, inSkipConn):  # x2是左侧的输出，x1是上一大层来的输出
        inData = self.Up(inData)

        diffY = inSkipConn.size()[2] - inData.size()[2]
        diffX = inSkipConn.size()[3] - inData.size()[3]

        inData = F.pad(inData, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([inSkipConn, inData], dim=1) # 将两个tensor拼接在一起 dim=1：在通道数（C）上进行拼接
        return self.Output(x)

# 定义最终的输出
class OutputConv(nn.Module):
    def __init__(self, inInputChannels, inOutputChannels):
        super(OutputConv, self).__init__()
        
        self.Blocks = nn.Conv2d(inInputChannels, inOutputChannels, 1)
    def forward(self, inData):
        return self.Blocks(inData)

class UNet2D(nn.Module):
    def __init__(self, inNumChannels, inNumClasses): # in_channels 图片的通道数，1为灰度图，3为彩色图
        super(UNet2D, self).__init__()
        self.NumChannels = inNumChannels
        self.NumClasses = inNumClasses

        self.InConv = InputConv(inNumChannels, 64)
        self.Down1 = DownSample(64, 128)
        self.Down2 = DownSample(128, 256)
        self.Down3 = DownSample(256, 512)
        self.Down4 = DownSample(512, 512)
        self.Up1 = UpSample(1024, 256)
        self.Up2 = UpSample(512, 128)
        self.Up3 = UpSample(256, 64)
        self.Up4 = UpSample(128, 64)
        self.OutConv = OutputConv(64, inNumClasses)

    def forward(self, inData):

        x1 = self.InConv(inData)
        x2 = self.Down1(x1)
        x3 = self.Down2(x2)
        x4 = self.Down3(x3)
        x5 = self.Down4(x4)
        out = self.Up1(x5, x4)
        out = self.Up2(out, x3)
        out = self.Up3(out, x2)
        out = self.Up4(out, x1)

        return self.OutConv(out)



class UNet2DNew(UNetBase):
    def __init__(self, inInputDim, inOutputDim, inImageSize, inLevelCount) -> None:
        super().__init__(inInputDim, inOutputDim, inImageSize, inLevelCount, InputConv, DownSample, DoubleConv, DownSample, OutputConv)

