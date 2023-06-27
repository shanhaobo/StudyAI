import torch
from torch import nn
from einops import rearrange
from torch import einsum

from Modules.Attention2D import IMultiHeadAttention2D


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        attv = self.to_qkv(x)
        qkv = attv.chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        print("q:{},  k:{}".format(q.size(), k.size()))
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        
        print("att:{},  v:{}".format(attn.size(), v.size()))
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        print("out pre:{}".format(out.size()))
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        print("out pos:{}".format(out.size()))
        return self.to_out(out)


att = Attention(3)

x = torch.randn((1, 3, 100, 200))

y = att(x)
print(y.size())

iatt = IMultiHeadAttention2D(3, 4, 32)
z = iatt.CalcAtt(x)
print(z.size())
