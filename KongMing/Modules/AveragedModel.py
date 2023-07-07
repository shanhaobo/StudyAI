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
    def __init__(self, model, device=None, avg_fn=None, use_buffers=False):
        super(AveragedModel, self).__init__()
        self.module = deepcopy(model)
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

    def update_parameters(self, model):
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers else model.parameters()
        )
        for p_swa, p_model in zip(self_param, model_param):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_,
                                                 self.n_averaged.to(device)))
        self.n_averaged += 1

    def override_parameters(self, model):
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers else model.parameters()
        )
        for p_swa, p_model in zip(self_param, model_param):
            with torch.no_grad():
                p_model.data.copy_(p_swa.data)
