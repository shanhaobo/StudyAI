import torch

from .BaseTrainer import BaseTrainer

class MultiNNTrainer(BaseTrainer) :
    def __init__(
            self,
            inLearningRate,
            inLogRootPath
        ) -> None:
        super().__init__(
            inLearningRate,
            inLogRootPath
        )

        pass
