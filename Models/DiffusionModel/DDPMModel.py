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

from Archiver.BaseArchiver import BaseArchiver
from Models.BaseModel import BaseModel
from Trainer.BaseTrainer import BaseTrainer

class DDPMModel(BaseModel) :
    def __init__(self, inTrainer: BaseTrainer, inArchiver: BaseArchiver):
        super().__init__(inTrainer, inArchiver)

    