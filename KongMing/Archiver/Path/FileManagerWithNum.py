from .BaseFileManager import BaseFileManager
import os
import re

class FileManagerWithNum(BaseFileManager) :
    def __init__(self, inRootDir: str, inExtension:str, inRange, inNeedTimestampDir:bool = False) -> None:
        super().__init__(inRootDir, inExtension, inNeedTimestampDir)
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
    
    def GetAllLeafDirNames(self, inPath = None) :
        TPath = inPath
        if not inPath :
            TPath = self.RawRootPath

        AllSubDirNames = [d for d in os.listdir(TPath) if (os.path.isdir(os.path.join(TPath, d)))]
        AllLeafDirNames = []
        for string in AllSubDirNames:
           if (re.match(r'(\d+)_(\d+)', string)) :
                AllLeafDirNames.append(string)

        if len(AllLeafDirNames) == 0 :
            return None
        
        return AllLeafDirNames
    
