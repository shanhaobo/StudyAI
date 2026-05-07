import torch
from .BaseModelFactory import BaseModelFactory

from KongMing.Archiver.MultiNNArchiver import MultiNNArchiver
from KongMing.Trainer.MultiNNTrainer import MultiNNTrainer

from KongMing.Utils.Executor import ResolveModelTag

from typing import Dict as TypedDict
from typing import List as TypedList

class MultiNNModelFacotry(BaseModelFactory):
    def __init__(
            self,
            inMultiNNDict : TypedDict[str, torch.nn.Module],
            inTrainer : MultiNNTrainer,
            inModelRootFolderPath : str,
            inNNModuleNameOnlyForTrain : TypedList[str] = None,
            inModelTag : str = None
        ):
        self.MultiNNDict : TypedDict[str, torch.nn.Module] = {}

        # tag 优先级：构造参数 > CLI --ModelTag= > 无
        Tag = inModelTag if inModelTag is not None else ResolveModelTag()

        # new Archiver
        NewArchiver = MultiNNArchiver(
            inModelRootFolderPath,
            inNNModuleNameOnlyForTrain,
            inModelTag=Tag
        )
        # set Log Root Path
        inTrainer.LogRootPath = NewArchiver.GetCurrTrainRootPath()

        super().__init__(inTrainer, NewArchiver)

        for Name, NN in inMultiNNDict.items():
            self.MultiNNDict[Name] = NN.to(self.Device)

        self.Trainer.RegisterMultiNNModule(self.MultiNNDict)
        self.Archiver.RegisterMultiNNModule(self.MultiNNDict)


# 旧名 MultiNNModelFacotry 是历史拼写错误；保留 alias 让新代码用正确拼写
MultiNNModelFactory = MultiNNModelFacotry
