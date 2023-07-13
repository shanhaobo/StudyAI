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

    def ApplyLossFunc(self, inLossFunc, **inKVArgs):
        if inspect.isclass(inLossFunc):
            self._LossFunction = inLossFunc(**inKVArgs)
        elif inspect.isfunction(inLossFunc) or inspect.ismethod(inLossFunc):
            self._LossFunction = inLossFunc
        else:
            raise TypeError

    def ApplyLoss(self, inInput: torch.Tensor, inTarget: torch.Tensor, **inKVArgs):
        self._Loss = self._LossFunction(inInput, inTarget, **inKVArgs)
        self._AvgLoss.AcceptNewValue(self._Loss.item())

    def GetLoss(self):
        return self._Loss.item(), self._AvgLoss.item()

    def ApplyEMA(self, inModule : Optional['BaseNNModule'], inDecay):
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
        
