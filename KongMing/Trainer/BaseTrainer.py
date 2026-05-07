import abc
import torch

from torch import Tensor
from torch.optim.optimizer import Optimizer

from torch.utils.data import DataLoader

from KongMing.Utils.Delegate import Delegate

class BaseTrainer(abc.ABC):

    #####

    class Backpropagate():
        def __init__(self, inOptimizer : Optimizer) -> None:
            self.Optimizer = inOptimizer
            self.Loss = None

        def Apply(self, inLoss : Tensor) -> None:
            self.Loss = inLoss

        def __enter__(self):
            self.Optimizer.zero_grad()
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            self.Loss.backward()
            self.Optimizer.step()

    #####

    def __init__(self, inLearningRate) -> None:
        self.Device             = None
        
        self.LearningRate       = inLearningRate

        self.BeginTrain         = Delegate()
        # End* 链通常挂 IO（Save / log close）；用 isolate 模式收集异常但不中断后续 callback
        # （比如某次 Save 抛了，后面的 log 关闭也要执行；scheduler.step 已在更前面跑过）
        self.EndTrain           = Delegate(bIsolateFailure=True)

        self.BeginEpochTrain    = Delegate()
        self.EndEpochTrain      = Delegate(bIsolateFailure=True)

        self.BeginBatchTrain    = Delegate()
        self.EndBatchTrain      = Delegate()

        # Validation 链：可选；BaseTrainer 默认 _BatchValid 是空操作。
        # 子类按需重写 _BatchValid 并把 metric 累到自己的成员上，再在 EndEpochValid hook 里打印。
        self.BeginEpochValid    = Delegate()
        self.EndEpochValid      = Delegate(bIsolateFailure=True)
        self.BeginBatchValid    = Delegate()
        self.EndBatchValid      = Delegate()

        self.CurrEpochIndex     = 0
        self.CurrBatchIndex     = 0

        self.BatchNumPerEpoch   = 0
        self.BatchNumPerValid   = 0

        self.EndEpochIndex      = 0

        self.SoftExit           = False

        self.LogRootPath        = "."

        self.PrintInterval      = 1     # 每 N 个 batch 打印一次；通过 inKVArgs["PrintInterval"] 覆盖

    def ShouldPrintBatch(self) -> bool:
        """子类的 EndBatchTrain hook 在 print 前检查；最后一个 batch 总是 print"""
        bIsLastBatch = (self.CurrBatchIndex + 1) == self.BatchNumPerEpoch
        return bIsLastBatch or ((self.CurrBatchIndex + 1) % self.PrintInterval == 0)

    @staticmethod
    def _BackPropagate(inOptimizer : Optimizer, inLoss : Tensor) -> None:
        inOptimizer.zero_grad()
        inLoss.backward()
        inOptimizer.step()
    
    @staticmethod
    def _BeginBackPropagate(inOptimizer : Optimizer) -> None:
        inOptimizer.zero_grad()
    
    @staticmethod
    def _EndBackPropagate(inOptimizer : Optimizer, inLoss : Tensor) -> None:
        inLoss.backward()
        inOptimizer.step()
    
    @abc.abstractmethod
    def _CreateOptimizer(self) -> None:
        pass

    @abc.abstractmethod
    def _CreateLossFN(self) -> None:
        pass

    @abc.abstractmethod
    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :
        pass

    def _BatchValid(self, inBatchData, inBatchLabel, inArgs, inKVArgs) -> None:
        """子类按需重写。默认空实现 = "传 inValidLoader 但子类没实现 valid"，等价于不验证。"""
        pass

    def __DontOverride__EpochValid(self, inValidLoader : DataLoader, inArgs, inKVArgs) -> None:
        # Eval 模式 + no_grad：BatchNorm/Dropout 行为正确，不更新参数也不留梯度
        self.BeginEpochValid(inArgs, inKVArgs)

        DataLen = len(inValidLoader.dataset)
        BatchSize = inValidLoader.batch_size
        self.BatchNumPerValid = -(-DataLen // BatchSize)

        with torch.no_grad():
            for self.CurrBatchIndex, (CurrBatchData, CurrBatchLabel) in enumerate(inValidLoader):
                self.BeginBatchValid(inArgs, inKVArgs)
                self._BatchValid(CurrBatchData, CurrBatchLabel, inArgs, inKVArgs)
                self.EndBatchValid(inArgs, inKVArgs)

        self.EndEpochValid(inArgs, inKVArgs)

    def __DontOverride__EpochTrain(self, inDataLoader:DataLoader, inArgs, inKVArgs) -> None:
        # Begin Epoch Train 
        # call BeginEpochTrain
        self.BeginEpochTrain(inArgs, inKVArgs)

        DataLen = len(inDataLoader.dataset)
        BatchSize = inDataLoader.batch_size
        #self.BatchNumPerEpoch = (DataLen // BatchSize)  + 0 if (DataLen % BatchSize == 0) else 1
        self.BatchNumPerEpoch = -(-DataLen // BatchSize)
        # For Each Batch Train
        for self.CurrBatchIndex, (CurrBatchData, CurrBatchLabel) in enumerate(inDataLoader):
            self.BeginBatchTrain(inArgs, inKVArgs)
            self._BatchTrain(CurrBatchData, CurrBatchLabel, inArgs, inKVArgs)
            self.EndBatchTrain(inArgs, inKVArgs)

        # End Epoch Train
        # call EndEpochTrain
        self.EndEpochTrain(inArgs, inKVArgs)


    def __DontOverride__Train(self, inDataLoader:DataLoader, inStartEpochIndex : int, inEpochIterCount : int, inArgs, inKVArgs, inValidLoader : DataLoader = None) -> None:
        # Begin Train
        # Create Optimizer & Loss Function
        self._CreateOptimizer()
        self._CreateLossFN()
        self.BeginTrain(inArgs, inKVArgs)

        self.CurrEpochIndex = inStartEpochIndex
        self.EndEpochIndex = (self.CurrEpochIndex + inEpochIterCount) if (inEpochIterCount > 0) else 0
        while self.__Continue_EpochIterCount():
            self.__DontOverride__EpochTrain(inDataLoader, inArgs, inKVArgs)
            # 验证：只在 inValidLoader 存在时跑一遍。__DontOverride__EpochValid 内部
            # 已经包了 no_grad，BN/Dropout 由 _BatchValid 子类自己 model.eval() 切换。
            if inValidLoader is not None:
                self.__DontOverride__EpochValid(inValidLoader, inArgs, inKVArgs)
            if self.SoftExit or self._CheckEndEpoch():
                break
            self.CurrEpochIndex += 1

        # End Train
        self.EndTrain(inArgs, inKVArgs)

    def Train(self, inDataLoader : DataLoader, inStartEpochIndex : int, inEpochIterCount : int, inArgs, inKVArgs, inValidLoader : DataLoader = None) -> None:
        if inStartEpochIndex < 0:
            inStartEpochIndex = 0
        self.__DontOverride__Train(inDataLoader, inStartEpochIndex, inEpochIterCount, inArgs, inKVArgs, inValidLoader=inValidLoader)

    def _CheckEndEpoch(self)->bool:
        return False
    
    def __Continue_EpochIterCount(self) -> bool:
        if self.EndEpochIndex <= 0:
            return True
        
        return self.CurrEpochIndex < self.EndEpochIndex
