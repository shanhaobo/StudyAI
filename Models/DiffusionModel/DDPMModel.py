import torch
from torch.utils.data import DataLoader
from torch import nn, einsum
import torch.nn.functional as F

from datetime import datetime

import math
from inspect import isfunction
from functools import partial

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from Models.BaseModel import BaseModel

from Archiver.DDPMArchiver import DDPMArchiver
from Trainer.DDPMTrainer import DDPMTrainer

import torch.nn.functional as F

from .Utils import BetaSchedule

from ..Moduels.UNet2D import UNet2D

from .DiffusionModelUtils import ConditionUNet

from .DiffusionModelBase import DiffusionModelBase

class DMModel(ConditionUNet):
    def __init__(self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            with_time_emb=True,
            resnet_block_groups=8,
            use_convnext=True,
            convnext_mult=2
        ) -> None:
        super().__init__(dim, init_dim, out_dim, dim_mults, channels, with_time_emb, resnet_block_groups, use_convnext, convnext_mult)


class DDPMModel(BaseModel) :
    def __init__(self, inImageSize, inChannel, inLearningRate=0.00001, inTimesteps : int = 1000, inModeRootlFolderPath="."):

        self.NNModel    = DMModel(dim=inImageSize, channels=inChannel, dim_mults=(1,2,4,))
        self.DiffusionModel = DiffusionModelBase(inTimesteps=inTimesteps)
        NewTrainer      = DDPMTrainer(self.NNModel, self.DiffusionModel, inLearningRate, inTimesteps=inTimesteps)
        NewArchiver     = DDPMArchiver(self.NNModel, self.DiffusionModel, inModeRootlFolderPath)
        super().__init__(NewTrainer, NewArchiver)

    def Eval(self, *inArgs, **inKWArgs):
        if (super().Eval(*inArgs, **inKWArgs) == False) :
            return None
        self.NNModel.eval()
        return self.NNModel.Sample(
            self.NNModel,
            inImageSize=inKWArgs["inImageSize"],
            inBatchSize=inKWArgs["inBatchSize"],
            inChannels=inKWArgs["inChannels"]
        )
