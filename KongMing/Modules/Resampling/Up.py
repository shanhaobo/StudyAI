from torch import nn

class Up_2(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()

        self.Blocks = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            # kernel_size=3, stride=1, padding=1 保持输入大小变
            nn.Conv2d(inInputDim, inInputDim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, inData):
        return self.Blocks(inData)

class ConvT_2(nn.Module):
    def __init__(self, inInputDim) -> None:
        super().__init__()

        self.Blocks = nn.ConvTranspose2d(inInputDim, inInputDim, kernel_size=4, stride=2, padding=1)

    def forward(self, inData):
        return self.Blocks(inData)

