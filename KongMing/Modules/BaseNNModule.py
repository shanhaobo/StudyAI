import torch

import inspect

from KongMing.Utils.AveragedUtils import EMAValue

from .AveragedModel import EMA as EMAModle

from typing import Optional

class BaseNNModule(torch.nn.Module):
    
    _Optimizer : torch.optim.Optimizer  = None

    _LossFunction                       = None
    _Loss : torch.Tensor                = None
    _AvgLoss : EMAValue                 = EMAValue(0.99)
    
    EMA : EMAModle                      = None
    EMAHolder : EMAModle                = None

    def __init__(self) -> None:
        super().__init__()

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
        if inModule is None:
            self.EMA = None
            self.EMAHolder = EMAModle(self, inDecay)
        else:
            self.EMA = EMAModle(inModule, inDecay)
            inModule.EMAHolder = self.EMA

    def __enter__(self):
        self._Optimizer.zero_grad()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._Loss.backward()
        self._Optimizer.step()
        if (self.EMAHolder is not None):
            self.EMAHolder.update_parameters(self)
