import torch

from .MultiNNTrainer import MultiNNTrainer

class CodecTrainer(MultiNNTrainer) :
    def __init__(self, 
            inEncoder : torch.nn.Module,
            inDecoder : torch.nn.Module,
            inLearningRate,
            inLogRootPath
        ) -> None:
        super().__init__(inLearningRate, inLogRootPath)
        self.Encoder = inEncoder
        self.Decoder = inDecoder

    def _CreateOptimizer(self) -> None:
        pass
    
    def _CreateLossFN(self) -> None:
        pass

    def _BatchTrain(self, inBatchData, inBatchLabel, inArgs, inKVArgs) :
        # get BatchSize
        nBatchSize = inBatchData.size(0)
        
        # Prepare Real and Fake Data
        RealData = inBatchData.to(self.Device)
        
        with self.Decoder as D:
            with self.Encoder as E:
                pass
            pass
