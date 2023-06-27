from KongMing.Modules.AveragedModel import AveragedModel

class EMA(AveragedModel):
    def __init__(self, model, decay, device="cpu"):
        super().__init__(model, device)
        self.decay = decay
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)
