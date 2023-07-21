from .BaseArchiver import BaseArchiver

class MultiNNArchiver(BaseArchiver):
    def __init__(
            self,
            inNNModelDict,
            inModelRootFolderPath : str
        ) -> None:
        super().__init__(inModelRootFolderPath)

        for Name, NNModel in enumerate(inNNModelDict):
            self.NNModelDict[Name] = NNModel
