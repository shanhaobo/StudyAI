import os
import torch
from KongMing.Utils.ModelFileOp import FindFileWithMaxNum

from .Path.FileManagerWithNum import FileManagerWithNum

from typing import Dict as TypedDict
from typing import List as TypedList

class BaseArchiver(object):
    def __init__(self, inModelRootFolderPath : str, inNNModuleNameOnlyForTrain : TypedList[str] = None) -> None:
        self.ModelArchiveRootFolderPath = os.path.join(inModelRootFolderPath, "ArchivedModels")

        self.FileNameManager            = FileManagerWithNum(self.ModelArchiveRootFolderPath, ".pkl", 100, True)

        self.SaveEpochIndex             = -1
        self.NNModuleDict : TypedDict[str, torch.nn.Module] = {}
        # 默认参数不能直接写 [] —— 那是 mutable default，所有实例共享同一个 list
        self.NNModuleNameOnlyForTrain   = inNNModuleNameOnlyForTrain if inNNModuleNameOnlyForTrain is not None else []

############################################################################
    def GetCurrTrainRootPath(self):
        return self.FileNameManager.MakeAndGetRootPath()
############################################################################

    def IsExistModel(self) -> bool:
        for Name, _ in self.NNModuleDict.items():
            # 训练专用模块（如 GAN 的 D）即使没存过也不该让 IsExistModel 返回 False
            # ——它们没有持久化语义，只在训练循环内活着。
            if Name in self.NNModuleNameOnlyForTrain:
                continue
            Path, _ = self.FindLatestModelFile(Name)
            if Path is None:
                return False

        return True

############################################################################

    def Eval(self):
        # 旧实现 del NNModuleDict[Name] 会让"Eval 之后再 inc"丢掉训练模块，是单向操作。
        # 改为切到 eval 模式 + 移到 cpu 释放显存；NNModuleDict 注册保持完整。
        for Name in self.NNModuleNameOnlyForTrain:
            Module = self.NNModuleDict.get(Name)
            if Module is None:
                continue
            Module.eval()
            try:
                Module.cpu()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

############################################################################

    def MakeNeuralNetworkArchiveFullPath(self, inNeuralNetworkName : str, inEpochIndex : int) -> str:
        return self.FileNameManager.MakeFileFullPathAndFileName(FileName = inNeuralNetworkName, Num = inEpochIndex)

    def GetFileFromValidLatestTimestampDirPath(self, inNeuralNetworkName : str, inEpochIndex : int) -> str:
        return self.FileNameManager.GetFilePathAndNameFromTimestampDirPathByEpoch(FileName = inNeuralNetworkName, Num = inEpochIndex)

    def GetLatestModelFolder(self) -> str :
        _, LatestLeafFolderPath, _ = self.FileNameManager.GetValidLatestTimestampDirInfo()

        return LatestLeafFolderPath

    def FindLatestModelFile(self, inModelName : str):
        LatestFolderPath = self.GetLatestModelFolder()
        if LatestFolderPath is None :
            return None, None

         # 返回数字最大（也就是最新）的文件
        FileName, MaxNum =  FindFileWithMaxNum(os.listdir(LatestFolderPath), inModelName, "*", "pkl")
        if FileName is None :
            return None, None

        return os.path.join(LatestFolderPath, FileName), MaxNum

############################################################################

    def Save(self, inEpochIndex : int) -> None:
        # if SaveEpochIndex == inEpochIndex means already saved
        if (self.SaveEpochIndex < inEpochIndex):
            self._Save(inEpochIndex=inEpochIndex)
            self.SaveEpochIndex = inEpochIndex

    def _Save(self, inEpochIndex : int) -> None:
        # 原子保存：先把所有模块写到 .tmp，全部成功后再 rename。
        # 避免中途断电/Ctrl+C 留下"半保存"epoch（GAN 存了 G 没存 D）。
        WrittenTmpPaths : TypedList = []
        try:
            for Name, Model in self.NNModuleDict.items():
                ModelFolderPath, ModelFileName = self.MakeNeuralNetworkArchiveFullPath(Name, inEpochIndex)
                os.makedirs(ModelFolderPath, exist_ok=True)
                ModelFullPath = os.path.join(ModelFolderPath, ModelFileName)
                TmpPath = ModelFullPath + ".tmp"
                # BaseNNModel 走 archive 协议（带 Optimizer / LRScheduler 状态）；
                # 其它普通 nn.Module 退回旧的 state_dict。
                if hasattr(Model, "StateDictForArchive"):
                    torch.save(Model.StateDictForArchive(), TmpPath)
                else:
                    torch.save(Model.state_dict(), TmpPath)
                WrittenTmpPaths.append((TmpPath, ModelFullPath))

            for TmpPath, ModelFullPath in WrittenTmpPaths:
                os.replace(TmpPath, ModelFullPath)
                print("Save Model:" + ModelFullPath)
        except Exception:
            # 任一失败：清理已写的 .tmp，避免污染目录
            for TmpPath, _ in WrittenTmpPaths:
                if os.path.exists(TmpPath):
                    try:
                        os.remove(TmpPath)
                    except OSError:
                        pass
            raise

    def Load(self, inEpochIndex : int):
        for Name, _ in self.NNModuleDict.items():
            FilePath, FileName = self.GetFileFromValidLatestTimestampDirPath(Name, inEpochIndex)
            if FilePath is None:
                return False
            ModelFullPath = os.path.join(FilePath, FileName)
            self.__LoadInto(self.NNModuleDict[Name], ModelFullPath)
            print("Load Model:" + ModelFullPath)

        return True

    @staticmethod
    def __LoadInto(inModule : torch.nn.Module, inFullPath : str) -> None:
        # 兼容新旧两种 checkpoint：BaseNNModel 走 archive 协议，普通 nn.Module 走 state_dict。
        # weights_only=True 是 PyTorch 2.6+ 的默认值，显式写出来防止 unpickle 任意类——
        # 我们存的内容只有 dict / OrderedDict / Tensor / Python 标量，纯白名单类型，没问题。
        Loaded = torch.load(inFullPath, weights_only=True)
        if hasattr(inModule, "LoadStateDictFromArchive"):
            inModule.LoadStateDictFromArchive(Loaded)
        else:
            # 旧 .pkl 是 state_dict；新格式万一被普通 nn.Module 撞上，取 "Model" 子项
            if isinstance(Loaded, dict) and ("Model" in Loaded):
                inModule.load_state_dict(Loaded["Model"])
            else:
                inModule.load_state_dict(Loaded)

    def LoadLastest(self):
        MaxEpochIndex = -1
        for Name, _ in self.NNModuleDict.items():
            EpochIndex = self.LoadLastestByModelName(Name)
            if EpochIndex is None:
                return None
            if EpochIndex > MaxEpochIndex :
                MaxEpochIndex = EpochIndex
        return MaxEpochIndex

    def LoadLastestByModelName(self, inModelName : str):
        ModelFullPath, EpochIndex = self.FindLatestModelFile(inModelName)
        if ModelFullPath is None :
            return None
        self.__LoadInto(self.NNModuleDict[inModelName], ModelFullPath)
        print("Load Model:" + ModelFullPath)
        return EpochIndex

    def LoadModelByTimestamp(self, inTimestamp:str, inEpochIndex):
        for Name, Model in self.NNModuleDict.items():
            ModelFullPath = self.FileNameManager.GetFilePathByTimestamp(
                inTimestamp=inTimestamp,
                Num=inEpochIndex,
                FileName=Name
            )
            if ModelFullPath is None :
                return None
            self.__LoadInto(Model, ModelFullPath)
            print("Load Model:" + ModelFullPath)

############################################################################
