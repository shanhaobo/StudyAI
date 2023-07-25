###################################
import os
OutputPath = "output/{}".format(os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(OutputPath, exist_ok=True)
###################################

import torch

lF1 = torch.nn.CrossEntropyLoss().to("cuda")
lF2 = torch.nn.CrossEntropyLoss().to("cuda")

model1 = torch.nn.Linear(128,  256).to("cuda")

model2 = torch.nn.Linear(256, 512).to("cuda")

opti1 = torch.optim.Adam(model1.parameters(), lr=0.0001, betas=(0.9, 0.999))
opti2 = torch.optim.Adam(model2.parameters(), lr=0.0001, betas=(0.9, 0.999))

output1 = model1(torch.randn(128).to("cuda"))
opti1.zero_grad()
l1 = lF1(output1, torch.randn_like(output1).to("cuda"))
l1.backward()
opti1.step()

output = model2(output1.detach())
opti2.zero_grad()
l2 = lF2(output, torch.randn_like(output).to("cuda"))
l2.backward()
opti2.step()




