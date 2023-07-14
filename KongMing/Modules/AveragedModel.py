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
    def __init__(self, inModel, device=None, avg_fn=None, use_buffers=False):
        super(AveragedModel, self).__init__()
        self.module = deepcopy(inModel)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + \
                    (model_parameter - averaged_model_parameter) / (num_averaged + 1)
        self.avg_fn = avg_fn
        self.use_buffers = use_buffers
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, inModel):
        self_param = (
            itertools.chain(self.module.named_parameters(), self.module.named_buffers())
            if self.use_buffers else self.named_parameters()
        )
        model_param = (
            itertools.chain(inModel.named_parameters(), inModel.named_buffers())
            if self.use_buffers else inModel.named_parameters()
        )
        model_param_dict = {name: param for name, param in model_param}
        for (swan, p_swa) in self_param:
            p_model = model_param_dict[swan]
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_,
                                                 self.n_averaged.to(device)))
        self.n_averaged += 1

    def override_parameters(self, inModel):
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers else self.parameters()
        )
        model_param = (
            itertools.chain(inModel.parameters(), inModel.buffers())
            if self.use_buffers else inModel.parameters()
        )
        for p_swa, p_model in zip(self_param, model_param):
            device = p_model.device
            p_swa_ = p_swa.detach().to(device)
            p_model.detach().copy_(p_swa_)

## ExponentialMovingAverage
class EMA(AveragedModel):
    def __init__(self, inModel, inDecay, device="cpu"):
        super().__init__(inModel, device)
        self.decay = inDecay
        def ema_avg(avg_model_param, model_param, num_averaged):
            return inDecay * avg_model_param + (1 - inDecay) * model_param

        super().__init__(inModel, device, ema_avg, use_buffers=True)
