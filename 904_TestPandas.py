###################################
import os
OutputPath = "output/{}".format(os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(OutputPath, exist_ok=True)
###################################

import torch

Alpha = torch.rand(128, 2, 3, 4, out=None)
print(Alpha.size())

