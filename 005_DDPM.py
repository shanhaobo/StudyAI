import torch
import torchvision

from datetime import datetime

from torchvision.utils import save_image
import os

from KongMing.Models.DiffusionModel.DDPMModel import DDPMModel

from KongMing.Utils.Executor import Executor

torch.set_printoptions(precision=10, sci_mode=False)


from torchvision import transforms
from torch.utils.data import DataLoader
#from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

image_size = 64
image_channel = 3
batch_size = 16
EmbedDim = 32
if __name__ == "__main__" :
    DDPM = DDPMModel(
        inEmbeddingDim=EmbedDim,
        inColorChanNum= image_channel,
        inLearningRate=0.00001,
        inTimesteps=1000,
        inModeRootlFolderPath="./output/005_DDPM/trained_models/"
    )
    Exec = Executor(DDPM)

    if Exec.IsExistModel() and Exec.ReadyTrain() == False:
        GenImage = Exec.Eval(
            inImageSize=image_size,
            inColorChanNum=image_channel,
            inBatchSize=15
        )
        print(GenImage.size())
        
        reverse_transform = transforms.Compose([
            transforms.Normalize((-0.5,), (2.0,)),
            transforms.Lambda(lambda t : (t + 1) * 0.5)
        ])
        Path = "output/005_DDPM/images"
        os.makedirs(Path, exist_ok=True)
        save_image(reverse_transform(GenImage), "{}/{}.png".format(Path, datetime.now().strftime("%Y%m%d%H%M%S")), nrow=5, normalize=True)
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(), # turn into Numpy array of shape HWC, divide by 255
            transforms.Lambda(lambda t : (t * 2) - 1),
            transforms.Normalize((0.5,), (0.5,))
        ])
        if False :
            dataset = torchvision.datasets.FashionMNIST(
                root="./data", train=True, transform=transform, download=True
            )
        else:
            dataset = torchvision.datasets.ImageFolder(root='D:/AI/Datasets/cartoon_faces', transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        Exec.Train(dataloader, SaveModelInterval=10)
