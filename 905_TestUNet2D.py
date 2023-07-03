###################################
import os
OutputPath = "output/{}".format(os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(OutputPath, exist_ok=True)
###################################

import torch

from KongMing.Modules.UNets.UNet2D import UNet2D_ConvNeXt, UNet2D_WSR
from torchvision.transforms import transforms

#input  dim 1
#output dim 8
#net = UNet2D_ConvNeXt(3, 32, [1, 2, 4])
net = UNet2D_WSR(3, 32, [1, 2, 4])

"D:/__DevAI__/Datasets/cartoon_faces/faces/00bfa209214d28bd4a22b64fa73841fb-0.jpg"

def SumParameters(inNN):
    return sum(p.nelement() for p in inNN.parameters())

s = SumParameters(net)
print("sum of params:{:,}".format(s))
OutputPath = "output/905_TestUNet2D"
os.makedirs(OutputPath, exist_ok=True)
with open("{}/tree.txt".format(OutputPath), 'w') as f:
    print(net, file=f)


t = torch.randn((1, 3, 64, 64))
v = torch.randint(0, 1000, (1,)).long()
x = net(t, v)

print(x.size())
