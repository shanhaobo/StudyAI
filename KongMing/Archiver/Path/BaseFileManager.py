import os
from datetime import datetime

#Root directory->Timestamp dirctory(optional)->Leaf directory(terminal directory)->File
class BaseFileManager() :
    def __init__(self, inRootPath:str, inExtension:str, inNeedTimestampDir:bool = False) -> None:
        self.RawRootPath    = inRootPath
        self.Extension      = inExtension if inExtension.startswith(".") else ("." + inExtension)
        os.makedirs(inRootPath, exist_ok=True)

        self.NeedJoinTimestampDir   = inNeedTimestampDir
        self.TimeStampDirFormat     = "%Y%m%d%H%M%S"

        self.__RootPath             = None

    def MakeLeafDirName(self, **inKWArgs) -> str:
        pass

    def MakeFileName(self, **inKWArgs) -> str:
        pass

    ################
    def GetFilePathByTimestamp(self, inTimestamp:str,  **inKWArgs)->str:
        LeafDir = self.MakeLeafDirName(**inKWArgs)
        FileName = self.MakeFileName(**inKWArgs)

        if LeafDir is None or FileName is None:
            return None

        TimeStampPath = os.path.join(self.RawRootPath, inTimestamp)
        LeafPath = os.path.join(TimeStampPath, LeafDir)

        return os.path.join(LeafPath, FileName)

    def MakeTimestampDirName(self)->str:
        return datetime.now().strftime(self.TimeStampDirFormat)
    
    def GetRootPath(self):
        if self.__RootPath is None :
            if self.NeedJoinTimestampDir : 
                self.__RootPath = os.path.join(self.RawRootPath, self.MakeTimestampDirName())
                os.makedirs(self.__RootPath, exist_ok=True)
                self.NeedJoinTimestampDir = False
            else:
                self.__RootPath = self.RawRootPath

        return self.__RootPath

    def MakeFileFullPath(self, **inKWArgs):
        RootPath = self.GetRootPath()

        LeafDirName = self.MakeLeafDirName(**inKWArgs)
        if not LeafDirName:
            return None

        LeafDirFullPath = os.path.join(RootPath, LeafDirName)
        os.makedirs(LeafDirFullPath, exist_ok=True)

        FileName = self.MakeFileName(**inKWArgs)
        if not FileName:
            return None
        
        if FileName.endswith(self.Extension) == False:
            FileName = FileName + self.Extension
        
        return os.path.join(LeafDirFullPath, FileName)
    
    def GetAllTimestampDirNames(self) :
        AllSubDirNames = [d for d in os.listdir(self.RawRootPath) if (os.path.isdir(os.path.join(self.RawRootPath, d)))]
        AllTimestampDirs = []
        for string in AllSubDirNames:
            try:
                datetime.strptime(string, self.TimeStampDirFormat)
                AllTimestampDirs.append(string)
            except ValueError:
                pass
        return AllTimestampDirs
    
    def GetLatestTimestampDirPath(self) :
        AllTimestampDirNames = self.GetAllTimestampDirNames()
        if len(AllTimestampDirNames) == 0:
            return None
        
        AllTimestampDirNames.sort(key=lambda x: int(x), reverse=True)

        return os.path.join(self.RawRootPath, AllTimestampDirNames[0])
