import torch

from .BaseTrainer import BaseTrainer

class SingleNNTrainer(BaseTrainer) :
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
