import torch
import torchvision

import numpy as np

from PIL import Image

from Models.DiffusionModel.DDPMModel import DDPMModel

torch.set_printoptions(precision=10, sci_mode=False)


from torchvision import transforms
from torch.utils.data import DataLoader
#from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

transform = transforms.Compose([
           transforms.RandomHorizontalFlip(), # 随机水平翻转图片
           transforms.ToTensor(), # 转成张量
           transforms.Lambda(lambda t: (t * 2) - 1) # 归一化到[-1,1]
])
image_size = 28
channels = 1
batch_size = 128
dataset = torchvision.datasets.FashionMNIST(
   root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

reverse_transform = transforms.Compose([
     transforms.Lambda(lambda t: (t + 1) / 2),
     transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     transforms.Lambda(lambda t: t * 255.),
     transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
     transforms.ToPILImage(),
])

model = DDPMModel(None, None, 10)
print("================")

def Extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    print(out.shape)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

t = torch.randint(2, 10, (4,)).long()
print(t)
x_start = torch.randn((3, 4, 5))
print(model.SqrtAlphasCumprod)
e = Extract(model.SqrtAlphasCumprod, t, x_start.shape)
print(e.shape)
print(e)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(), # turn into Numpy array of shape HWC, divide by 255
    transforms.Lambda(lambda t: (t * 2) - 1),
    
])

url = 'D:/AI/Datasets/cartoon_faces/faces/00a44dac107792065c96f27664e91cf6-0.jpg'
image = Image.open(url)

x_start = transform(image).unsqueeze(0)
x_start.shape
def get_noisy_image(x_start, t):
  # add noise
  x_noisy = model.DMModel.Q_Sample(x_start, t=t)

  # turn back into PIL image
  noisy_image = reverse_transform(x_noisy.squeeze())

  return noisy_image

t = torch.tensor([40])

get_noisy_image(x_start, t)
