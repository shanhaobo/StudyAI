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

    def _BatchTrain(self, inBatchData, inBatchLabel, *inArgs, **inKWArgs) :
        # get BatchSize
        nBatchSize = inBatchData.size(0)
        
        pass
