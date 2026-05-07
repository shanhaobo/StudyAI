import torch

import inspect

from KongMing.Utils.AveragedUtils import EMAValue

from ..Modules.AveragedModule import EMA as EMAModle

from typing import Optional

class BaseNNModel(torch.nn.Module):

    ##---------------------------------------##
    class BackPropagaterClass() :
        def __init__(self) -> None:
            super().__init__()

            self._Optimizer : torch.optim.Optimizer = None
            self._LRScheduler : torch.optim.lr_scheduler._LRScheduler = None

            self._LossFunction                      = None
            self._Loss                              = None
            # EMA decay 从 0.99 起步；BaseModelFactory 在 BeginTrain 时若发现
            # inKVArgs["LossEMADecay"] 会覆盖（avoid hardcoding for short-epoch runs）。
            self._AvgLoss : EMAValue                = EMAValue(0.99)

            # Pending optimizer/scheduler state restored from a checkpoint;
            # applied lazily once ApplyOptimizer/ApplyLRScheduler creates them.
            self._PendingState                      = None

            # ── AMP / 梯度累积 ───────────────────────────────────────
            # _AMPDtype = None    → 关闭 AMP（裸 fp32 走旧路径）
            # _AMPDtype = float16 → 走 GradScaler；bfloat16 → 走 autocast 不需要 scaler
            # _AccumSteps = 1     → 每个 batch 都 step；>1 时累 N 个 batch 才 step 一次
            self._AMPDtype       = None
            self._GradScaler     = None
            self._AccumSteps     = 1
            self._AccumCounter   = 0

        def ApplyAMP(self, inDtype) -> None:
            """开启混合精度。inDtype: torch.float16 / torch.bfloat16 / None"""
            self._AMPDtype = inDtype
            if inDtype is torch.float16:
                # fp16 容易下溢，需要 GradScaler 把 loss 放大再 backward
                self._GradScaler = torch.cuda.amp.GradScaler()
            else:
                # bf16 动态范围与 fp32 相当，不需要 scaler
                self._GradScaler = None

        def ApplyGradAccum(self, inSteps : int) -> None:
            self._AccumSteps = max(1, int(inSteps))
            self._AccumCounter = 0

        def IsAccumBoundary(self) -> bool:
            """当前是否到达累积步数边界（即真的 step 当前批次）"""
            return (self._AccumCounter % self._AccumSteps) == 0

        def ApplyOptimizer(self, inModule:torch.nn.Module, inOptimizerType, inLearningRate, **inKVArgs):
            if inspect.isclass(inOptimizerType):
                self._Optimizer = inOptimizerType(inModule.parameters(), inLearningRate, **inKVArgs)
            else:
                raise TypeError

            if self._Optimizer is None:
                raise RuntimeError

            self.__TryApplyPendingOptimizerState()

        def ApplyLRScheduler(self, inSchedulerType, **inKVArgs):
            if inspect.isclass(inSchedulerType):
                self._LRScheduler = inSchedulerType(self._Optimizer, **inKVArgs)
            else:
                raise TypeError

            if self._LRScheduler is None:
                raise RuntimeError

            self.__TryApplyPendingLRSchedulerState()

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
            # 累积模式下只在边界 zero_grad（保留中间累积的梯度）
            if self.IsAccumBoundary():
                self._Optimizer.zero_grad()

        def EndBackPropagate(self):
            if self._Loss is None:
                # 之前的逻辑就是静默 pass；保留行为，只是不再 step（避免空 step 撞坏 scheduler）
                return

            # accumulation：把每个累积步的 loss 均摊，等价于"大 batch"的平均梯度
            if self._AccumSteps > 1:
                LossForBackward = self._Loss / self._AccumSteps
            else:
                LossForBackward = self._Loss

            if self._GradScaler is not None:
                # fp16 路径
                self._GradScaler.scale(LossForBackward).backward()
            else:
                # fp32 / bf16 路径
                LossForBackward.backward()

            self._AccumCounter += 1
            if not self.IsAccumBoundary():
                # 累积中：只 backward 不 step，下一个 batch 继续累
                return

            if self._GradScaler is not None:
                self._GradScaler.step(self._Optimizer)
                self._GradScaler.update()
            else:
                self._Optimizer.step()

        def BackPropagate(self):
            self.BeginBackPropagate()
            self.EndBackPropagate()

        def UpdateLRScheduler(self):
            if self._LRScheduler is not None:
                self._LRScheduler.step()

        ##-- Archive helpers ---------------------------##
        def StateDict(self) -> dict:
            return {
                "Optimizer"   : self._Optimizer.state_dict()   if self._Optimizer   is not None else None,
                "LRScheduler" : self._LRScheduler.state_dict() if self._LRScheduler is not None else None,
            }

        def StashPendingState(self, inState : dict) -> None:
            # Optimizer / LRScheduler are usually created AFTER load (in _CreateOptimizer);
            # stash here and let Apply* drain it.
            self._PendingState = inState
            self.__TryApplyPendingOptimizerState()
            self.__TryApplyPendingLRSchedulerState()

        def __TryApplyPendingOptimizerState(self) -> None:
            if self._PendingState is None or self._Optimizer is None:
                return
            OptState = self._PendingState.get("Optimizer")
            if OptState is not None:
                self._Optimizer.load_state_dict(OptState)
                self._PendingState["Optimizer"] = None

        def __TryApplyPendingLRSchedulerState(self) -> None:
            if self._PendingState is None or self._LRScheduler is None:
                return
            SchState = self._PendingState.get("LRScheduler")
            if SchState is not None:
                self._LRScheduler.load_state_dict(SchState)
                self._PendingState["LRScheduler"] = None

    ##---------------------------------------##

    ##---------------------------------------##
    # EMA 注册表：键是 target nn.Module 的 id，值是跟踪它的 EMAModle。
    # 替代旧版的单一全局槽（EMAHolder/EMATargeModule），允许多个网络（如 GAN G/D）各自持有 EMA。
    # 不存到实例属性是为了避免被 nn.Module.__setattr__ 自动注册为子模块（self.EMA 已经是子模块了，
    # 再存一份 self.EMATargetModule = self 会造成 state_dict() 自指递归）。
    _EMARegistry : dict = {}

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

    def ApplyAMP(self, inDtype):
        self.BackPropagater.ApplyAMP(inDtype)

    def ApplyGradAccum(self, inSteps : int):
        self.BackPropagater.ApplyGradAccum(inSteps)

    ##---------------------------------------##
    def BackPropagate(self):
        self.BackPropagater.BackPropagate()

        self.__UpdateEMA()
    ##---------------------------------------##

    ##---------------------------------------##
    def __enter__(self):
        self.BackPropagater.BeginBackPropagate()
        # AMP：forward + loss 计算都包在 autocast 里；离开 with 块前手动退出
        AMPDtype = self.BackPropagater._AMPDtype
        if AMPDtype is not None and torch.cuda.is_available():
            self._AutocastCtx = torch.autocast(device_type="cuda", dtype=AMPDtype)
            self._AutocastCtx.__enter__()
        else:
            self._AutocastCtx = None
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # 先关 autocast，再 backward（backward 必须在 fp32 域里跑）
        if getattr(self, "_AutocastCtx", None) is not None:
            self._AutocastCtx.__exit__(exc_type, exc_value, traceback)
            self._AutocastCtx = None

        self.BackPropagater.EndBackPropagate()

        self.__UpdateEMA()
    ##---------------------------------------##

    ##---------------------------------------##
    def ApplyEMA(self, inDecay, inModule : Optional['BaseNNModel'] = None):
        TargetModule = inModule if inModule is not None else self

        # self.EMA 仍然是 nn.Module 子模块（保留 state_dict 旧形态），caller 可通过 self.EMA 访问；
        # 真正"哪个 target 触发更新"通过 _EMARegistry 查表，避免类级单槽被多网络互相覆盖。
        self.EMA = EMAModle(TargetModule, inDecay)
        BaseNNModel._EMARegistry[id(TargetModule)] = self.EMA

    def __UpdateEMA(self):
        # 当 self 是某 EMA 的 target 时（包括 self 给自己注册了 EMA 的常见情形），
        # 自动用当前参数更新对应 EMA。
        ema = BaseNNModel._EMARegistry.get(id(self))
        if ema is not None:
            ema.UpdateParameters(self)
    ##---------------------------------------##

    ##-- Archive helpers --------------------------------##
    def StateDictForArchive(self) -> dict:
        return {
            "Model"         : self.state_dict(),
            "BackPropagater": self.BackPropagater.StateDict(),
        }

    def LoadStateDictFromArchive(self, inLoaded) -> None:
        # 兼容两种 checkpoint：
        #   - 新格式 dict: {"Model": state_dict, "BackPropagater": {...}}
        #   - 旧格式: 直接是 state_dict (OrderedDict)
        if isinstance(inLoaded, dict) and ("Model" in inLoaded):
            self.load_state_dict(inLoaded["Model"])
            BPState = inLoaded.get("BackPropagater")
            if BPState is not None:
                self.BackPropagater.StashPendingState(BPState)
        else:
            self.load_state_dict(inLoaded)
    ##---------------------------------------------------##
