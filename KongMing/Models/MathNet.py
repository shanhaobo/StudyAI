import torch
from torch import nn

from KongMing.Models.BaseNNModel import BaseNNModel


class MathNet(BaseNNModel):
    def __init__(self, inNumOps : int = 3, inHiddenSize : int = 128) -> None:
        super().__init__()

        # 输入维度 = 2(操作数 a,b) + NumOps(op 的 one-hot)
        # one-hot 比把 op idx 当标量喂进来好得多：MLP 没有"序号"语义，
        # 标量输入会让网络误以为 op=2 离 op=0 更远，是常见踩坑点。
        InputSize = 2 + inNumOps

        self.Backbone = nn.Sequential(
            nn.Linear(InputSize, inHiddenSize),
            nn.ReLU(inplace=True),
            nn.Linear(inHiddenSize, inHiddenSize),
            nn.ReLU(inplace=True),
            nn.Linear(inHiddenSize, inHiddenSize),
            nn.ReLU(inplace=True),
            nn.Linear(inHiddenSize, 1),
        )

    def forward(self, inX : torch.Tensor) -> torch.Tensor:
        return self.Backbone(inX)
