from .BaseTrainer import BaseTrainer

class MultiNNTrainer(BaseTrainer) :
    def __init__(
            self,
            inMNNDict,
            inLearningRate,
            inLogRootPath
        ) -> None:
        super().__init__(
            inLearningRate,
            inLogRootPath
        )
        self.MNNDict = {}

        for NNName, NN in inMNNDict.items():
            self.MNNDict[NNName] = NN.to(self.Device)
