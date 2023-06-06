import torch

from .MultiNNTrainer import MultiNNTrainer

class CodecTrainer(MultiNNTrainer) :
    def __init__(self, 
            inEncoder : torch.nn.Module,
            inDecoder : torch.nn.Module,
            inLearningRate
        ) -> None:
        super().__init__(inLearningRate)
        self.Encoder = inEncoder
        self.Decoder = inDecoder

    def _BatchTrain(self, inBatchDatum, inBatchLabel, *inArgs, **inKWArgs) :
        # get BatchSize
        nBatchSize = inBatchDatum.size(0)
        
        pass
