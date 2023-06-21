import torch
import torchvision

from datetime import datetime

from torchvision.utils import save_image
import os

from Models.DiffusionModel.DDPMModel import DDPMModel

from Utils.Executor import Executor

torch.set_printoptions(precision=10, sci_mode=False)


from torchvision import transforms
from torch.utils.data import DataLoader
#from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

image_size = 32
channels = 1
batch_size = 16
if __name__ == "__main__" :
    DDPM = DDPMModel(inEmbedDims=image_size, inChannel= channels, inLearningRate=0.00001, inTimesteps=1000, inModeRootlFolderPath="./trained_models/DDPM")
    Exec = Executor(DDPM)

    if Exec.IsExistModel() and Exec.ReadyTrain() == False:
        GenImage = Exec.Eval(
            inImageSize=image_size,
            inBatchSize=1,
            inChannels=channels
        )
        print(GenImage.size())
        
        reverse_transform = transforms.Compose([
            transforms.Normalize((-0.5,), (2.0,)),
            transforms.Lambda(lambda t : (t + 1) * 0.5)
        ])
        Path = "images/DDPM"
        os.makedirs(Path, exist_ok=True)
        save_image(reverse_transform(GenImage), "{}/{}.png".format(Path, datetime.now().strftime("%Y%m%d%H%M%S")), nrow=5, normalize=True)
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(), # turn into Numpy array of shape HWC, divide by 255
            transforms.Lambda(lambda t : (t * 2) - 1),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, transform=transform, download=True
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        Exec.Train(dataloader, SaveModelInterval=1)

"""
url = 'D:/AI/Datasets/cartoon_faces/faces/00a44dac107792065c96f27664e91cf6-0.jpg'
image = Image.open(url)

x_start = transform(image).unsqueeze(0)
x_start.shape
def get_noisy_image(x_start, t):
  # add noise
  x_noisy = model.DMModel.Q_Sample(x_start, inT=t, inNoise=torch.randn_like(x_start))

  # turn back into PIL image
  noisy_image = reverse_transform(x_noisy.squeeze())

  return noisy_image

t = torch.tensor([2])

genimg = get_noisy_image(x_start, t)

save_image(transform(genimg), "images/{}.png".format(datetime.now().strftime("%Y%m%d%H%M%S")), nrow=5, normalize=True)
"""
