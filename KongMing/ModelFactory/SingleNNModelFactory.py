import torch
from .BaseModelFactory import BaseModelFactory

from KongMing.Archiver.SingleNNArchiver import SingleNNArchiver
from KongMing.Trainer.SingleNNTrainer import SingleNNTrainer

from KongMing.Models.BaseNNModel import BaseNNModel

class SingleNNModelFactory(BaseModelFactory):
    def __init__(
            self,
            inNNModel : BaseNNModel,
            inTrainer : SingleNNTrainer,
            inModelRootFolderPath : str
        ) :
        # new Archiver
        NewArchiver = SingleNNArchiver(
            inModelRootFolderPath
        )
        # set Log Root Path
        inTrainer.LogRootPath = NewArchiver.GetCurrTrainRootPath()

        super().__init__(inTrainer, NewArchiver)

        inTrainer.NNModel = inNNModel.to(self.Device)
        NewArchiver.NNModuleDict["NNModel"] = inTrainer.NNModel

