import os
import sys
import json
import torch

from datetime import datetime
from torch.utils.data import DataLoader

from KongMing.Utils.CaseInsensitiveContainer import CaseInsensitiveList, CaseInsensitiveDict

from KongMing.Trainer.BaseTrainer import BaseTrainer
from KongMing.Archiver.BaseArchiver import BaseArchiver


class _StdoutTee:
    """把 print 输出同步写到日志文件；不替换 stdout 全局，仅在 Begin/End Train 之间挂载。"""
    def __init__(self, inFile, inOriginal):
        self.File     = inFile
        self.Original = inOriginal

    def write(self, inText):
        self.Original.write(inText)
        try:
            self.File.write(inText)
            self.File.flush()
        except Exception:
            pass

    def flush(self):
        self.Original.flush()
        try:
            self.File.flush()
        except Exception:
            pass

    def __getattr__(self, inName):
        return getattr(self.Original, inName)

class BaseModelFactory(object):
    def __init__(self, inTrainer : BaseTrainer, inArchiver : BaseArchiver):
        if torch.cuda.is_available():
            self.Device = torch.device("cuda")
            print(torch.cuda.get_device_name(self.Device))
        else:
            self.Device = torch.device("cpu")
            print("CUDA unavailable, using CPU")

        self.Trainer        = inTrainer
        self.Archiver       = inArchiver

        self.Trainer.Device = self.Device
        
        self.Trainer.BeginTrain.add(self.__BMBeginTrain)
        self.Trainer.EndBatchTrain.add(self.__BMEndBatchTrain)
        self.Trainer.EndEpochTrain.add(self.__BMEndEpochTrain)
        self.Trainer.EndTrain.add(self.__BMEndTrain)

        self.ForceSave      = False

        self.SaveInterval   = 10

        # 日志：挂在 LogRootPath/train.log；BeginTrain 时打开，EndTrain 时关闭
        self._LogFile : object = None
        self._StdoutBackup     = None
        # 结构化指标日志：每个 print 间隔写一行 jsonl 到 train.metrics.jsonl
        # 后期画曲线直接读 jsonl 不用正则；和 train.log（人类可读）并存
        self._MetricsFile : object = None

    ###########################################################################################

    def NewTrain(self, inDataLoader : DataLoader, inEpochIterCount : int, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None, inValidLoader : DataLoader = None) -> None:
        self.Trainer.Train(inDataLoader, 0, inEpochIterCount, inArgs, inKVArgs, inValidLoader=inValidLoader)

    def IncTrain(self, inDataLoader : DataLoader, inStartEpochNum : int, inEpochIterCount : int, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None, inValidLoader : DataLoader = None) -> None:
        if inStartEpochNum >= 0 and self.Archiver.Load(inStartEpochNum):
            pass
        else:
            inStartEpochNum = self.Archiver.LoadLastest()

        if inStartEpochNum is None:
            self.Trainer.Train(inDataLoader, 0, inEpochIterCount, inArgs, inKVArgs, inValidLoader=inValidLoader)
        else:
            self.Trainer.Train(inDataLoader, inStartEpochNum + 1, inEpochIterCount, inArgs, inKVArgs, inValidLoader=inValidLoader)

    def LoadLastest(self, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None):
        EpochIndex = self.Archiver.LoadLastest()
        if (EpochIndex is None) or (EpochIndex < 0):
            return False
        self.Trainer.CurrEpochIndex = EpochIndex + 1
        return True
    
    def Load(self, inEpoch, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None):
        self.Archiver.Load(inEpoch)
        
    def IsExistModels(self) -> bool:
        return self.Archiver.IsExistModel()

    # alias: Executor 一直叫 IsExistModel（无 s），两边随便用
    IsExistModel = IsExistModels
    
    def Eval(self, inEpoch, inArgs : CaseInsensitiveList = None, inKVArgs : CaseInsensitiveDict = None):
        self.Archiver.Eval()
        if inEpoch <= 0:
            self.LoadLastest(inArgs, inKVArgs)
        else:
            self.Load(inEpoch)

    ###########################################################################################

    def _SumParameters(self,inNN):
        return sum(p.nelement() for p in inNN.parameters())

    ###########################################################################################

    def __BMBeginTrain(self, inArgs, inKVArgs)->None:
        self.__OpenLogFile()
        print("Begin Training... [{}]".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        # 自动登记 Trainer 上的 BaseNNModel 属性——给"新写 Trainer 子类只放 self.X = SomeNet()
        # 就想让 Archiver 自动持久化"的场景兜底。已登记的（按 id 去重）不会被重复登记，
        # 所以对现有 Single/Multi 入口零影响。
        self.__AutoRegisterNNModules()
        # 直接调 BaseModelFactory.NewTrain 不走 Executor 时 inKVArgs 可能为 None
        if inKVArgs is None:
            inKVArgs = CaseInsensitiveDict()
        SaveInterval = inKVArgs.get("SaveInterval")
        if SaveInterval is not None:
            self.SaveInterval = int(SaveInterval)
        PrintInterval = inKVArgs.get("PrintInterval")
        if PrintInterval is not None:
            self.Trainer.PrintInterval = int(PrintInterval)
        LossEMADecay = inKVArgs.get("LossEMADecay")
        if LossEMADecay is not None:
            # 把 loss 平均的 decay 推到所有 BaseNNModel 子模块，避免硬编码 0.99 在短 epoch 不合适
            from KongMing.Models.BaseNNModel import BaseNNModel
            Decay = float(LossEMADecay)
            for Module in self.Archiver.NNModuleDict.values():
                if isinstance(Module, BaseNNModel):
                    Module.BackPropagater._AvgLoss.Decay = Decay

        # ── AMP / 梯度累积 ──
        # --AMP=bf16 / fp16 / off ；--GradAccum=N
        from KongMing.Models.BaseNNModel import BaseNNModel as _BaseNN
        AMPArg = inKVArgs.get("AMP")
        AMPDtype = None
        if AMPArg is not None:
            AMPNorm = str(AMPArg).strip().lower()
            if AMPNorm in ("bf16", "bfloat16"):
                AMPDtype = torch.bfloat16
            elif AMPNorm in ("fp16", "float16", "half"):
                AMPDtype = torch.float16
            elif AMPNorm in ("off", "none", "fp32", "float32", ""):
                AMPDtype = None
            else:
                print("[BaseModelFactory] Unknown AMP value '{}', ignored".format(AMPArg))
        if AMPDtype is not None and not torch.cuda.is_available():
            print("[BaseModelFactory] AMP requested but CUDA unavailable, falling back to fp32")
            AMPDtype = None

        AccumArg = inKVArgs.get("GradAccum")
        AccumSteps = int(AccumArg) if AccumArg is not None else 1

        if AMPDtype is not None or AccumSteps > 1:
            print("[BaseModelFactory] AMP={} GradAccum={}".format(AMPDtype, AccumSteps))

        for Module in self.Archiver.NNModuleDict.values():
            if isinstance(Module, _BaseNN):
                if AMPDtype is not None:
                    Module.ApplyAMP(AMPDtype)
                if AccumSteps > 1:
                    Module.ApplyGradAccum(AccumSteps)

    ############################################

    def __BMEndBatchTrain(self, inArgs, inKVArgs) -> None:
        if self._MetricsFile is None:
            return
        # 跟随 PrintInterval：和屏幕打印同节奏，避免每 batch 都写硬盘
        if not self.Trainer.ShouldPrintBatch():
            return
        from KongMing.Models.BaseNNModel import BaseNNModel
        Record = {
            "Epoch" : self.Trainer.CurrEpochIndex,
            "Batch" : self.Trainer.CurrBatchIndex + 1,
            "BatchNum" : self.Trainer.BatchNumPerEpoch,
            "Time"  : datetime.now().isoformat(timespec="seconds"),
        }
        for Name, Module in self.Archiver.NNModuleDict.items():
            if isinstance(Module, BaseNNModel) and Module.BackPropagater._Loss is not None:
                Loss, Avg = Module.GetLossValue()
                Record["{}.Loss".format(Name)] = Loss
                Record["{}.AvgLoss".format(Name)] = Avg
        try:
            self._MetricsFile.write(json.dumps(Record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def __BMEndEpochTrain(self, inArgs, inKVArgs) -> None:
        # 文件 sentinel：跨平台替代 keyboard 热键
        # `touch <LogRootPath>/.save` 触发当前 epoch 末强制保存
        # `touch <LogRootPath>/.exit` 触发当前 epoch 末软退出
        # Linux/Mac/容器里没法挂全局热键时这是唯一办法；和 keyboard 共存，谁先到位谁触发
        self.__CheckSentinelFiles()

        if self.ForceSave or ((self.Trainer.CurrEpochIndex + 1) % self.SaveInterval == 0):
            if self.ForceSave :
                self.ForceSave = False
                print("Epoch:{} Force Save Models".format(self.Trainer.CurrEpochIndex))
            else:
                print("Epoch:{} Save Models".format(self.Trainer.CurrEpochIndex))
            self.Archiver.Save(self.Trainer.CurrEpochIndex)

    def __CheckSentinelFiles(self) -> None:
        LogDir = self.Trainer.LogRootPath
        if not LogDir:
            return
        # 删除信号文件后才触发：避免一个文件忘了删导致每 epoch 都重入
        SaveSignal = os.path.join(LogDir, ".save")
        if os.path.exists(SaveSignal):
            try:
                os.remove(SaveSignal)
            except OSError as e:
                print("[BaseModelFactory] Failed to remove .save sentinel:", e)
            else:
                print("[BaseModelFactory] .save sentinel detected → ForceSave")
                self.ForceSaveAtEndEpoch()

        ExitSignal = os.path.join(LogDir, ".exit")
        if os.path.exists(ExitSignal):
            try:
                os.remove(ExitSignal)
            except OSError as e:
                print("[BaseModelFactory] Failed to remove .exit sentinel:", e)
            else:
                print("[BaseModelFactory] .exit sentinel detected → SoftExit")
                self.ForceExitAtEndEpoch()

    def __BMEndTrain(self, inArgs, inKVArgs)->None:
        self.Archiver.Save(self.Trainer.CurrEpochIndex)
        print("End Train!!! [{}]".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.__CloseLogFile()

    def __AutoRegisterNNModules(self) -> None:
        """扫 self.Trainer 上的公开属性，把 BaseNNModel 实例补登记到 Archiver.NNModuleDict。

        - 按 id() 去重：已显式登记过的不会被重复添加（即便它在 Trainer 上有不同的属性名）
        - 名字冲突：若属性名已在 dict 里但指向不同实例，跳过（不覆盖既有登记）
        - 跳过 _ 前缀属性：Trainer 内部缓存型 BaseNNModel 应当用 _ 前缀显式 opt-out
        """
        from KongMing.Models.BaseNNModel import BaseNNModel

        Existing = {id(m) for m in self.Archiver.NNModuleDict.values() if m is not None}
        NewlyAdded = []

        for AttrName in dir(self.Trainer):
            if AttrName.startswith("_"):
                continue
            try:
                Value = getattr(self.Trainer, AttrName)
            except AttributeError:
                continue
            if not isinstance(Value, BaseNNModel):
                continue
            if id(Value) in Existing:
                continue
            if AttrName in self.Archiver.NNModuleDict:
                # 同名但指向不同实例——保守起见跳过，不覆盖
                print("[BaseModelFactory] auto-register skipped '{}' (name already taken)".format(AttrName))
                continue
            self.Archiver.NNModuleDict[AttrName] = Value
            Existing.add(id(Value))
            NewlyAdded.append(AttrName)

        if NewlyAdded:
            print("[BaseModelFactory] auto-registered NN modules: {}".format(NewlyAdded))

    def __OpenLogFile(self) -> None:
        try:
            LogDir = self.Trainer.LogRootPath if self.Trainer.LogRootPath else "."
            os.makedirs(LogDir, exist_ok=True)
            LogPath = os.path.join(LogDir, "train.log")
            self._LogFile = open(LogPath, "a", encoding="utf-8", buffering=1)
            self._StdoutBackup = sys.stdout
            sys.stdout = _StdoutTee(self._LogFile, self._StdoutBackup)
        except Exception as e:
            print("[BaseModelFactory] Open log file failed, continue without tee:", e)
            self._LogFile = None
        # 结构化指标文件：失败不影响训练
        try:
            LogDir = self.Trainer.LogRootPath if self.Trainer.LogRootPath else "."
            os.makedirs(LogDir, exist_ok=True)
            MetricsPath = os.path.join(LogDir, "train.metrics.jsonl")
            self._MetricsFile = open(MetricsPath, "a", encoding="utf-8", buffering=1)
        except Exception as e:
            print("[BaseModelFactory] Open metrics file failed, continue without metrics:", e)
            self._MetricsFile = None

    def __CloseLogFile(self) -> None:
        if self._StdoutBackup is not None:
            sys.stdout = self._StdoutBackup
            self._StdoutBackup = None
        if self._LogFile is not None:
            try:
                self._LogFile.close()
            except Exception:
                pass
            self._LogFile = None
        if self._MetricsFile is not None:
            try:
                self._MetricsFile.close()
            except Exception:
                pass
            self._MetricsFile = None

    ###########################################################################################

    def ForceSaveAtEndEpoch(self) -> None:
        print("Accept Force Save.............")
        self.ForceSave = True

    def ForceExitAtEndEpoch(self) -> None:
        print("Accept Soft Exit.............")
        self.Trainer.SoftExit = True

    ###########################################################################################
