import torch
from torch.nn import Module

from copy import deepcopy

import itertools

class AveragedModule(Module):
    """
    You can also use custom averaging functions with `avg_fn` parameter.
    If no averaging function is provided, the default is to compute
    equally-weighted average of the weights.
    """
    def __init__(self, inModel, inDevice=None, inAvgFN=None, inUseBuffers=False):
        super(AveragedModule, self).__init__()
        self.Module = deepcopy(inModel)
        if inDevice is not None:
            self.Module = self.Module.to(inDevice)
        self.register_buffer('Averaged', torch.tensor(0, dtype=torch.long, device=inDevice))
        if inAvgFN is None:
            def inAvgFN(AvgModelParam, ModelParam, NumAvg):
                return AvgModelParam + (ModelParam - AvgModelParam) / (NumAvg + 1)
        self.AvgFN = inAvgFN
        self.UseBuffers = inUseBuffers
        
    def forward(self, *inArgs, **inKVargs):
        return self.Module(*inArgs, **inKVargs)

    def UpdateParameters(self, inModel):
        SelfParam = (
            itertools.chain(self.Module.named_parameters(), self.Module.named_buffers())
            if self.UseBuffers else self.named_parameters()
        )
        ModelParam = (
            itertools.chain(inModel.named_parameters(), inModel.named_buffers())
            if self.UseBuffers else inModel.named_parameters()
        )
        ModelParamDict = {MPName : MPParam for MPName, MPParam in ModelParam}
        for (SWAName, SWAParam) in SelfParam:
            ModelParam = ModelParamDict[SWAName]
            Device = SWAParam.device
            DeviceParamModel = ModelParam.detach().to(Device)
            if self.Averaged == 0:
                SWAParam.detach().copy_(DeviceParamModel)
            else:
                AvgValue = self.AvgFN(SWAParam.detach(), DeviceParamModel, self.Averaged.to(Device))
                SWAParam.detach().copy_(AvgValue)
        self.Averaged += 1

    def OverrideParameters(self, inModel):
        SelfParam = (
            itertools.chain(self.Module.parameters(), self.Module.buffers())
            if self.UseBuffers else self.parameters()
        )
        ModelParam = (
            itertools.chain(inModel.parameters(), inModel.buffers())
            if self.UseBuffers else inModel.parameters()
        )
        for SWAParam, ModelParam in zip(SelfParam, ModelParam):
            DeviceSWAParam = SWAParam.detach().to(ModelParam.device)
            ModelParam.detach().copy_(DeviceSWAParam)

## ExponentialMovingAverage
class EMA(AveragedModule):
    def __init__(self, inModel, inDecay, inDevice="cpu"):
        super().__init__(inModel, inDevice)
        self.decay = inDecay
        def EMA_Avg(AvgModelParam, ModelParam, NumAveraged):
            return inDecay * AvgModelParam + (1 - inDecay) * ModelParam

        super().__init__(inModel, inDevice, EMA_Avg, inUseBuffers=True)
