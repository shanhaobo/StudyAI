from torchvision.transforms import transforms
from torchvision.utils import save_image

from PIL import Image

from datetime import datetime

ImageURL = 'D:/AI/Datasets/cartoon_faces/faces/00a44dac107792065c96f27664e91cf6-0.jpg'
Img = Image.open(ImageURL)
Transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t : (t * 2) - 1),
    transforms.Normalize((0.5,), (0.5,))
])
Img1 = Transform1(Img)

Transform2 = transforms.Compose([
    transforms.Normalize((-0.5,), (2.0,)),
    transforms.Lambda(lambda t : (t + 1) * 0.5)
])
Img2 = Transform2(Img1)

now = datetime.now()
timestamp = now.strftime("%Y%m%d%H%M%S")
save_image(Img2, "images/{}.png".format(timestamp), nrow=5, normalize=True)


