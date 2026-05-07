import os
import sys
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
        pass

    def __BMEndEpochTrain(self, inArgs, inKVArgs) -> None:
        if self.ForceSave or ((self.Trainer.CurrEpochIndex + 1) % self.SaveInterval == 0):
            if self.ForceSave :
                self.ForceSave = False
                print("Epoch:{} Force Save Models".format(self.Trainer.CurrEpochIndex))
            else:
                print("Epoch:{} Save Models".format(self.Trainer.CurrEpochIndex))
            self.Archiver.Save(self.Trainer.CurrEpochIndex)

    def __BMEndTrain(self, inArgs, inKVArgs)->None:
        self.Archiver.Save(self.Trainer.CurrEpochIndex)
        print("End Train!!! [{}]".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.__CloseLogFile()

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

    ###########################################################################################

    def ForceSaveAtEndEpoch(self) -> None:
        print("Accept Force Save.............")
        self.ForceSave = True

    def ForceExitAtEndEpoch(self) -> None:
        print("Accept Soft Exit.............")
        self.Trainer.SoftExit = True

    ###########################################################################################
