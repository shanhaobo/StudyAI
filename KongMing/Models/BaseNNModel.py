import torch

import inspect

from KongMing.Utils.AveragedUtils import EMAValue

from ..Modules.AveragedModel import EMA as EMAModle

from typing import Optional

class BaseNNModel(torch.nn.Module):

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

    def ApplyEMA(self, inDecay, inModule : Optional['BaseNNModel'] = None):
        TargetModule = inModule
        if TargetModule is None:
            TargetModule = self

        self.EMA = EMAModle(TargetModule, inDecay)

        BaseNNModel.EMAHolder = self.EMA
        BaseNNModel.EMATargeModule = TargetModule

    def BackPropagate(self):
        self._Optimizer.zero_grad()
        self._Loss.backward()
        self._Optimizer.step()

        self.__UpdateEMA()

    def __UpdateEMA(self):
        if ((BaseNNModel.EMAHolder is not None) and (BaseNNModel.EMATargeModule == self)):
            BaseNNModel.EMAHolder.UpdateParameters(BaseNNModel.EMATargeModule)

    def __enter__(self):
        self._Optimizer.zero_grad()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._Loss is not None:
            self._Loss.backward()
        else:
            # 如果这里错误, 先屏蔽,看看哪里报错了
            # 因为这里不应该为None
            raise RuntimeError
        
        loss = self._Optimizer.step()
        print("exit:loss:{}".format(loss))

        self.__UpdateEMA()

