import torch
from torch.utils.data import DataLoader, Dataset

from dataclasses import dataclass

from KongMing.Utils.Executor import Executor
from KongMing.Utils.OutputPath import BuildOutputPath
from KongMing.Utils.ConfigUtils import ApplyConfigFromKV

from KongMing.ModelFactory.MathNet.MathNetModelFactory import MathNetModelFactory


###################################################################################################

# 三个 op：加 / 减 / 乘
# 去掉 power（值域可达 10^200，回归头被它支配）和 divide（除零 / 大值溢出）。
# 想让网络真的"学会算术"，必须保证目标值的尺度可控。
OpNames = ["add", "sub", "mul"]
NumOps  = len(OpNames)

# 输入归一到 [-1, 1]：a/b 缩放到 [0, 1]，再线性映射；输出按 InputScale^2 反归一回原始尺度。
InputScale = 100.0


@dataclass
class TrainConfig:
    DatasetLen      : int    = 200000
    EvalLen         : int    = 2000
    EpochCnt        : int    = 1
    BatchSize       : int    = 320
    HiddenSize      : int    = 128
    LearningRate    : float  = 0.001
    SaveInterval    : int    = 1
    PrintInterval   : int    = 50

Config = TrainConfig()
Overridden = set(ApplyConfigFromKV(Config))


###################################################################################################

class MathDataset(Dataset):
    """合成算术数据集。
    输入张量 5 维：[a_norm, b_norm, op_onehot(3)]；标签为标量（已按 InputScale^2 归一）。
    """

    def __init__(self, inLength : int) -> None:
        self.Length = inLength

    def __len__(self) -> int:
        return self.Length

    def __getitem__(self, inIdx : int):
        # 用 torch 自身随机数，避免依赖 Python random，方便 num_workers
        AB = torch.rand(2) * InputScale            # [0, 100)
        OpIdx = int(torch.randint(0, NumOps, ()).item())

        A, B = AB[0].item(), AB[1].item()
        if OpIdx == 0:
            Y = A + B
        elif OpIdx == 1:
            Y = A - B
        else:
            Y = A * B

        OpOneHot = torch.zeros(NumOps)
        OpOneHot[OpIdx] = 1.0

        Feature = torch.cat([AB / InputScale, OpOneHot], dim=0)
        Label   = torch.tensor([Y / (InputScale * InputScale)], dtype=torch.float32)
        return Feature, Label


###################################################################################################

if __name__ == "__main__":
    if Overridden:
        print("[Config] CLI overrides:", sorted(Overridden))

    Factory = MathNetModelFactory(
        inNumOps=NumOps,
        inHiddenSize=Config.HiddenSize,
        inLearningRate=Config.LearningRate,
        inModelRootFolderPath=BuildOutputPath(__file__, "Synthetic"),
    )
    Exec = Executor(Factory)

    DoEval = (Exec.ForceTrain() == False) and Exec.IsExistModel()

    if DoEval:
        EvalLoader = DataLoader(
            MathDataset(Config.EvalLen),
            batch_size=Config.BatchSize,
            shuffle=False,
        )
        Exec.Eval(
            inDataLoader=EvalLoader,
            inOpNames=OpNames,
            inInputScale=InputScale,
        )
    else:
        TrainLoader = DataLoader(
            MathDataset(Config.DatasetLen),
            batch_size=Config.BatchSize,
            shuffle=True,
        )
        Exec.Train(
            TrainLoader,
            SaveInterval=Config.SaveInterval,
            PrintInterval=Config.PrintInterval,
        )
