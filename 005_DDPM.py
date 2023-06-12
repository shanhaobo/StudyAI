import torch

from Models.DiffusionModel.DDPMModel import DDPMModel

torch.set_printoptions(precision=10, sci_mode=False)

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

