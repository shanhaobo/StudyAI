import torch

import inspect

from KongMing.Utils.AveragedUtils import EMAValue

from .AveragedModel import EMA as EMAModle

from typing import Optional

class BaseNNModule(torch.nn.Module):

    EMAHolder : EMAModle                = None
    EMATargeModule : torch.nn.Module    = None

    def __init__(self) -> None:
        super().__init__()

        self._Optimizer : torch.optim.Optimizer = None

        self._LossFunction                      = None
        self._Loss                              = None
        self._AvgLoss : EMAValue                = EMAValue(0.99)
    
        self.EMA : EMAModle                     = None

    def ApplyOptimizer(self, inOptimizerType, inLearningRate, **inKVArgs):
        if inspect.isclass(inOptimizerType):
            self._Optimizer = inOptimizerType(self.parameters(), inLearningRate, **inKVArgs)
        else:
            raise TypeError
        
        if self._Optimizer is None:
            raise RuntimeError

    def ApplyLossFunc(self, inLossFunc, **inKVArgs):
        if inspect.isclass(inLossFunc):
            self._LossFunction = inLossFunc(**inKVArgs)
        elif inspect.isfunction(inLossFunc) or inspect.ismethod(inLossFunc):
            self._LossFunction = inLossFunc
        elif callable(inLossFunc):
            self._LossFunction = inLossFunc
        else:
            raise TypeError

        if self._LossFunction is None:
            raise RuntimeError

    def AcceptLoss(self, inLoss: torch.Tensor):
        self._Loss = inLoss
        self._AvgLoss.AcceptNewValue(self._Loss.item())

    def CalcLoss(self, inInput: torch.Tensor, inTarget: torch.Tensor, **inKVArgs):
        return self._LossFunction(inInput, inTarget, **inKVArgs)
    
    def CalcAndAcceptLoss(self, inInput: torch.Tensor, inTarget: torch.Tensor = None, **inKVArgs):
        self._Loss = self._LossFunction(inInput, inTarget, **inKVArgs)
        self._AvgLoss.AcceptNewValue(self._Loss.item())

    def GetLossValue(self):
        return self._Loss.item(), self._AvgLoss.item()

    def ApplyEMA(self, inDecay, inModule : Optional['BaseNNModule'] = None):
        TargetModule = inModule
        if TargetModule is None:
            TargetModule = self

        self.EMA = EMAModle(TargetModule, inDecay)

        BaseNNModule.EMAHolder = self.EMA
        BaseNNModule.EMATargeModule = TargetModule

    def __enter__(self):
        self._Optimizer.zero_grad()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._Loss is not None:
            self._Loss.backward()
        else:
            # 如果这里错误, 先屏蔽,看看哪里报错了
            # 因为这里不应该为Non
            raise RuntimeError
        
        self._Optimizer.step()

        if ((BaseNNModule.EMAHolder is not None) and (BaseNNModule.EMATargeModule == self)):
            BaseNNModule.EMAHolder.update_parameters(BaseNNModule.EMATargeModule)
