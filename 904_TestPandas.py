###################################
import os
OutputPath = "output/{}".format(os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(OutputPath, exist_ok=True)
###################################

import torch

class Lo():
    LoFN = None
    def ApplyBCE(self, inBCE):
        self.LoFN = inBCE


l = Lo()
l2 = Lo()

l.ApplyBCE(torch.nn.BCELoss().to("cuda"))
l2.ApplyBCE(torch.nn.BCELoss().to("cuda"))

if l.LoFN is None:
    print("error")

if l.LoFN == l2.LoFN:
    print("same")

print(l.LoFN)
print(l2.LoFN)


