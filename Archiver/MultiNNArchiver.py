import torch

from .BaseArchiver import BaseArchiver

class MultiNNArchiver(BaseArchiver):
    def __init__(self, inModelPrefix : str, inModelRootFolderPath : str = ".", inNeedTimestamp : bool = True) -> None:
        super().__init__(inModelPrefix, inModelRootFolderPath, inNeedTimestamp)
        self.NNModelDict = {}

    def Save(self, inEpochIndex : int, inSuffix, inExtension) -> None:
        for Name, Model in self.NNModelDict.items():
            ModelFullPath = self.MakeNeuralNetworkArchiveFullPath(Name, inEpochIndex, inSuffix, inExtension)
            torch.save(Model.state_dict(), ModelFullPath)
            print("Save Model:" + ModelFullPath)

    def LoadLastest(self, inForTrain : bool = True):
        MaxEpochIndex = -1
        for Name, _ in self.NNModelDict.items():
            bSuccess, EpochIndex = self.LoadLastestByModelName(Name)
            if bSuccess == False :
                return False, -1
            if EpochIndex > MaxEpochIndex :
                MaxEpochIndex = EpochIndex
        return True, MaxEpochIndex

    def LoadLastestByModelName(self, inModelName : str):
        ModelFullPath, EpochIndex = self.FindLatestModelFile(inModelName)
        if ModelFullPath == None :
            return False, -1
        self.NNModelDict[inModelName].load_state_dict(torch.load(ModelFullPath))
        print("Load Model:" + ModelFullPath)
        return True, EpochIndex
