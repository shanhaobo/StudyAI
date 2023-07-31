import torch

import inspect

from KongMing.Utils.AveragedUtils import EMAValue

from ..Modules.AveragedModule import EMA as EMAModle

from typing import Optional

class BaseNNModel(torch.nn.Module):
    ##---------------------------------------##
    EMAHolder : EMAModle                = None
    EMATargeModule : torch.nn.Module    = None
    ##---------------------------------------##

    ##---------------------------------------##
    class BackPropagaterClass() :
        def __init__(self) -> None:
            super().__init__()

            self._Optimizer : torch.optim.Optimizer = None
            self._LRScheduler : torch.optim.lr_scheduler._LRScheduler = None

            self._LossFunction                      = None
            self._Loss                              = None
            self._AvgLoss : EMAValue                = EMAValue(0.99)
        
        def ApplyOptimizer(self, inModule:torch.nn.Module, inOptimizerType, inLearningRate, **inKVArgs):
            if inspect.isclass(inOptimizerType):
                self._Optimizer = inOptimizerType(inModule.parameters(), inLearningRate, **inKVArgs)
            else:
                raise TypeError
            
            if self._Optimizer is None:
                raise RuntimeError

        def ApplyLRScheduler(self, inSchedulerType, **inKVArgs):
            if inspect.isclass(inSchedulerType):
                self._LRScheduler = inSchedulerType(self._Optimizer, **inKVArgs)
            else:
                raise TypeError
            
            if self._LRScheduler is None:
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

        def BeginBackPropagate(self):
            self._Optimizer.zero_grad()

        def EndBackPropagate(self):
            if self._Loss is not None:
                self._Loss.backward()
            else:
                # 如果这里错误, 先屏蔽,看看哪里报错了
                # 因为这里不应该为None
                # raise RuntimeError
                pass

            self._Optimizer.step()

        def BackPropagate(self):
            self.BeginBackPropagate()
            self.EndBackPropagate()

        def UpdateLRScheduler(self):
            if self._LRScheduler is not None:
                self._LRScheduler.step()

    ##---------------------------------------##

    ##---------------------------------------##
    def __init__(self) -> None:
        super().__init__()

        self.BackPropagater = BaseNNModel.BackPropagaterClass()

        self.EMA : EMAModle = None

    def ApplyOptimizer(self, inOptimizerType, inLearningRate, **inKVArgs):
        self.BackPropagater.ApplyOptimizer(self, inOptimizerType, inLearningRate, **inKVArgs)

    def ApplyLRScheduler(self, inSchedulerType, **inKVArgs):
        self.BackPropagater.ApplyLRScheduler(inSchedulerType, **inKVArgs)

    def ApplyLossFunc(self, inLossFunc, **inKVArgs):
        self.BackPropagater.ApplyLossFunc(inLossFunc, **inKVArgs)

    def AcceptLoss(self, inLoss: torch.Tensor):
        self.BackPropagater.AcceptLoss(inLoss)

    def CalcLoss(self, inInput: torch.Tensor, inTarget: torch.Tensor, **inKVArgs):
        return self.BackPropagater.CalcLoss(inInput, inTarget, **inKVArgs)
    
    def CalcAndAcceptLoss(self, inInput: torch.Tensor, inTarget: torch.Tensor = None, **inKVArgs):
        self.BackPropagater.CalcAndAcceptLoss(inInput, inTarget, **inKVArgs)

    def GetLossValue(self):
        return self.BackPropagater.GetLossValue()

    def UpdateLRScheduler(self):
        self.BackPropagater.UpdateLRScheduler()

    ##---------------------------------------##
    def BackPropagate(self):
        self.BackPropagater.BackPropagate()
        
        self.__UpdateEMA()
    ##---------------------------------------##

    ##---------------------------------------##
    def __enter__(self):
        self.BackPropagater.BeginBackPropagate()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.BackPropagater.EndBackPropagate()

        self.__UpdateEMA()
    ##---------------------------------------##

    ##---------------------------------------##
    def ApplyEMA(self, inDecay, inModule : Optional['BaseNNModel'] = None):
        TargetModule = inModule
        if TargetModule is None:
            TargetModule = self

        self.EMA = EMAModle(TargetModule, inDecay)

        BaseNNModel.EMAHolder = self.EMA
        BaseNNModel.EMATargeModule = TargetModule

    def __UpdateEMA(self):
        if ((BaseNNModel.EMAHolder is not None) and (BaseNNModel.EMATargeModule == self)):
            BaseNNModel.EMAHolder.UpdateParameters(BaseNNModel.EMATargeModule)
    ##---------------------------------------##
