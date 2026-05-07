import torch
from .BaseModelFactory import BaseModelFactory

from KongMing.Archiver.SingleNNArchiver import SingleNNArchiver
from KongMing.Trainer.SingleNNTrainer import SingleNNTrainer

from KongMing.Models.BaseNNModel import BaseNNModel

from KongMing.Utils.Executor import ResolveModelTag

class SingleNNModelFactory(BaseModelFactory):
    def __init__(
            self,
            inNNModel : BaseNNModel,
            inTrainer : SingleNNTrainer,
            inModelRootFolderPath : str,
            inModelTag : str = None
        ) :
        # tag 优先级：构造参数 > CLI --ModelTag= > 无
        # 不传 tag 时路径与旧版 100% 一致；传了就在 ArchivedModels/ 下加一层子目录
        Tag = inModelTag if inModelTag is not None else ResolveModelTag()

        # new Archiver
        NewArchiver = SingleNNArchiver(
            inModelRootFolderPath,
            inModelTag=Tag
        )
        # set Log Root Path
        inTrainer.LogRootPath = NewArchiver.GetCurrTrainRootPath()

        super().__init__(inTrainer, NewArchiver)

        inTrainer.NNModel = inNNModel.to(self.Device)
        NewArchiver.NNModuleDict["NNModel"] = inTrainer.NNModel
