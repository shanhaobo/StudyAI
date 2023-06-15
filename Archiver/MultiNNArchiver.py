import torch

from .BaseArchiver import BaseArchiver

class MultiNNArchiver(BaseArchiver):
    def __init__(self, inModelPrefix : str, inModelRootFolderPath : str = ".") -> None:
        super().__init__(inModelPrefix, inModelRootFolderPath)
        self.NNModelDict = {}

    def _Save(self, inEpochIndex : int) -> None:
        for Name, Model in self.NNModelDict.items():
            ModelFullPath = self.MakeNeuralNetworkArchiveFullPath(Name, inEpochIndex)
            torch.save(Model.state_dict(), ModelFullPath)
            print("Save Model:" + ModelFullPath)

    def LoadLastest(self, inForTrain : bool = True):
        MaxEpochIndex = -1
        for Name, _ in self.NNModelDict.items():
            EpochIndex = self.LoadLastestByModelName(Name)
            if EpochIndex <= 0 :
                return None
            if EpochIndex > MaxEpochIndex :
                MaxEpochIndex = EpochIndex
        return MaxEpochIndex

    def LoadLastestByModelName(self, inModelName : str):
        ModelFullPath, EpochIndex = self.FindLatestModelFile(inModelName)
        if ModelFullPath is None :
            return None
        self.NNModelDict[inModelName].load_state_dict(torch.load(ModelFullPath))
        print("Load Model:" + ModelFullPath)
        return EpochIndex
