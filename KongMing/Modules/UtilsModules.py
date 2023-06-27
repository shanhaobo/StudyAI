from torch import nn

class DoubleLinearModule(nn.Module):
    def __init__(self, inInputDim, inOutputDim, inMidScale = 4) -> None:
        super().__init__()
        MidDim = inOutputDim * inMidScale
        self.Blocks = nn.Sequential(
            nn.Linear(inInputDim, MidDim),
            nn.GELU(),
            nn.Linear(MidDim, inOutputDim)
        )

    def forward(self, inData):
        return self.Blocks(inData)
