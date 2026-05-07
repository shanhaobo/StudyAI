import torch

from KongMing.ModelFactory.SingleNNModelFactory import SingleNNModelFactory
from KongMing.Trainer.MathNetTrainer import MathNetTrainer
from KongMing.Models.MathNet import MathNet

from KongMing.Utils.CaseInsensitiveContainer import CaseInsensitiveList, CaseInsensitiveDict


class MathNetModelFactory(SingleNNModelFactory):
    def __init__(
            self,
            inNumOps : int,
            inHiddenSize : int,
            inLearningRate : float,
            inModelRootFolderPath : str,
        ) -> None:
        self.MathNet = MathNet(inNumOps=inNumOps, inHiddenSize=inHiddenSize)

        Trainer = MathNetTrainer(inLearningRate)

        super().__init__(self.MathNet, Trainer, inModelRootFolderPath)

        print("Sum of Params:{:,} ".format(self._SumParameters(self.MathNet)))

    def Eval(self, inEpoch, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None) :
        if (super().Eval(inEpoch, inArgs, inKVArgs) == False) :
            return False

        EvalLoader = inKVArgs.get("inDataLoader") if inKVArgs is not None else None
        if EvalLoader is None:
            print("[MathNetModelFactory] Eval: missing inDataLoader")
            return False

        OpNames = inKVArgs.get("inOpNames") if inKVArgs is not None else None
        InputScale = float(inKVArgs.get("inInputScale", 100.0)) if inKVArgs is not None else 100.0

        self.MathNet.eval()

        TotalSE  = 0.0
        TotalAE  = 0.0
        Count    = 0
        Samples  = []
        SampleLimit = 10

        with torch.no_grad():
            for Batch, Label in EvalLoader:
                Batch  = Batch.to(self.Device)
                Label  = Label.to(self.Device)
                Pred   = self.MathNet(Batch).view(-1)
                Truth  = Label.view(-1)

                TotalSE += torch.sum((Pred - Truth) ** 2).item()
                TotalAE += torch.sum(torch.abs(Pred - Truth)).item()
                Count   += Truth.numel()

                if len(Samples) < SampleLimit:
                    # 反归一化打印更直观
                    A = Batch[:, 0] * InputScale
                    B = Batch[:, 1] * InputScale
                    OpOneHot = Batch[:, 2:]
                    OpIdx = torch.argmax(OpOneHot, dim=1)
                    DenormPred  = Pred  * (InputScale * InputScale)
                    DenormTruth = Truth * (InputScale * InputScale)
                    for i in range(Batch.shape[0]):
                        if len(Samples) >= SampleLimit:
                            break
                        Samples.append((
                            A[i].item(),
                            B[i].item(),
                            OpIdx[i].item(),
                            DenormPred[i].item(),
                            DenormTruth[i].item(),
                        ))

        MSE = TotalSE / max(Count, 1)
        MAE = TotalAE / max(Count, 1)

        print("---- MathNet Eval ----")
        print("Samples (denormalized):")
        for A, B, Op, P, T in Samples:
            OpName = OpNames[Op] if OpNames is not None and Op < len(OpNames) else str(Op)
            print("  {:>8.3f} {:^4} {:>8.3f}  pred={:>12.4f}  gt={:>12.4f}  err={:>+10.4f}".format(
                A, OpName, B, P, T, P - T
            ))
        print("Normalized   MSE: {:.6f} | MAE: {:.6f}".format(MSE, MAE))
        return True
