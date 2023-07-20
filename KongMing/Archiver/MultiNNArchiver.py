from .BaseArchiver import BaseArchiver

class MultiNNArchiver(BaseArchiver):
    def __init__(
            self,
            inNNModelDict,
            inModelPrefix : str,
            inModelRootFolderPath : str
        ) -> None:
        super().__init__(inModelPrefix, inModelRootFolderPath)

        for Name, NNModel in inNNModelDict:
            self.NNModelDict[Name] = NNModel
