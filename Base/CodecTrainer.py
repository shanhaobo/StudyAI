import torch

from .MultiNNTrainer import MultiNNTrainer

class CodecTrainer(MultiNNTrainer) :
    def __init__(self, inLearningRate) -> None:
        super().__init__(inLearningRate)
