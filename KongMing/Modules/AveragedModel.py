import torch
from torch.nn import Module

from copy import deepcopy

import itertools

class AveragedModel(Module):
    """
    You can also use custom averaging functions with `avg_fn` parameter.
    If no averaging function is provided, the default is to compute
    equally-weighted average of the weights.
    """
    def __init__(self, inModel, inDevice=None, inAvgFN=None, inUseBuffers=False):
        super(AveragedModel, self).__init__()
        self.Module = deepcopy(inModel)
        if inDevice is not None:
            self.Module = self.Module.to(inDevice)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=inDevice))
        if inAvgFN is None:
            def inAvgFN(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + \
                    (model_parameter - averaged_model_parameter) / (num_averaged + 1)
        self.AvgFN = inAvgFN
        self.UseBuffers = inUseBuffers
        self.Averaged = 0
        
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
        ModelParamDict = {name : param for name, param in ModelParam}
        for (SWAName, SWAParam) in SelfParam:
            ModelParam = ModelParamDict[SWAName]
            Device = SWAParam.device
            DeviceParamModel = ModelParam.detach().to(Device)
            if self.Averaged == 0:
                SWAParam.detach().copy_(DeviceParamModel)
            else:
                SWAParam.detach().copy_(self.AvgFN(SWAParam.detach(), DeviceParamModel,
                                                 self.Averaged.to(Device)))
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
class EMA(AveragedModel):
    def __init__(self, inModel, inDecay, inDevice="cpu"):
        super().__init__(inModel, inDevice)
        self.decay = inDecay
        def ema_avg(avg_model_param, model_param, num_averaged):
            return inDecay * avg_model_param + (1 - inDecay) * model_param

        super().__init__(inModel, inDevice, ema_avg, inUseBuffers=True)
