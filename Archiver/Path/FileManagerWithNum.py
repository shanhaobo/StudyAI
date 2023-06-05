from .BaseFileManager import BaseFileManager
import os
import re

class FileManagerWithNum(BaseFileManager) :
    def __init__(self, inRootDir: str, inExtension:str, inRange) -> None:
        super().__init__(inRootDir, inExtension)
        self.Range = inRange

    def MakeLeafDirName(self, **inKWArgs) -> str:
        Num = inKWArgs["Num"]
        if not Num:
            return None
        
        PreRange = (Num // self.Range) * self.Range
        PostRange = PreRange + self.Range

        return "{:0>6d}_{:0>6d}".format(PreRange, PostRange)

    def MakeFileName(self, **inKWArgs) -> str:
        FileName = inKWArgs["FileName"]
        if not FileName:
            return None

        Num = inKWArgs["Num"]
        if not Num:
            return None
        
        return "{}_{:0>6d}".format(FileName, Num)
    
    def GetAllLeafDirNames(self) :
        AllLeafDirNames = [d for d in os.listdir(self.RootDir) if (m := re.match(r'(\d+)_(\d+)', d))]
        AllLeafDirNames.sort(key=lambda x: int(x))
        return AllLeafDirNames
    
