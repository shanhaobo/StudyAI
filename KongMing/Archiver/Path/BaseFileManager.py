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
    
    def MakeAndGetRootPath(self):
        if self.__RootPath is None :
            if self.NeedJoinTimestampDir : 
                self.__RootPath = os.path.join(self.RawRootPath, self.MakeTimestampDirName())
                self.NeedJoinTimestampDir = False
            else:
                self.__RootPath = self.RawRootPath

        return self.__RootPath

    def MakeFileFullPath(self, **inKWArgs):
        RootPath = self.MakeAndGetRootPath()

        LeafDirName = self.MakeLeafDirName(**inKWArgs)
        if not LeafDirName:
            return None

        LeafDirFullPath = os.path.join(RootPath, LeafDirName)

        FileName = self.MakeFileName(**inKWArgs)
        if not FileName:
            return None
        
        if FileName.endswith(self.Extension) == False:
            FileName = FileName + self.Extension
        
        return os.path.join(LeafDirFullPath, FileName)
    
    def MakeFileFullPathAndFileName(self, **inKWArgs):
        RootPath = self.MakeAndGetRootPath()

        LeafDirName = self.MakeLeafDirName(**inKWArgs)
        if not LeafDirName:
            return None

        LeafDirFullPath = os.path.join(RootPath, LeafDirName)

        FileName = self.MakeFileName(**inKWArgs)
        if not FileName:
            return None
        
        if FileName.endswith(self.Extension) == False:
            FileName = FileName + self.Extension
        
        return LeafDirFullPath, FileName
    
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
    
    def GetAllLeafDirNames(self, inPath = None) :
        pass
    
    def TravelValidLatestTimestampDirInfo(self, ModelFilesFunc) :
        AllTimestampDirNames = self.GetAllTimestampDirNames()
        if len(AllTimestampDirNames) == 0:
            return None
        
        AllTimestampDirNames.sort(key=lambda x: int(x), reverse=True)

        for DirName in AllTimestampDirNames:
            LatestTimestampDirPath = os.path.join(self.RawRootPath, DirName)
            # 获取所有子文件夹
            LeafFolders = self.GetAllLeafDirNames(LatestTimestampDirPath)
            if LeafFolders is None:
                continue

            LeafFolders.sort(key=lambda x: int(x), reverse=True)

            for SF in LeafFolders:
                # 取最新的子文件夹
                LatestLeafFolderPath = os.path.join(LatestTimestampDirPath, SF)

                # 使用 glob 以及文件名前缀来获取子文件夹下所有的 .pkl 文件
                ModelFiles = os.listdir(LatestLeafFolderPath)

                if ModelFilesFunc(ModelFiles):
                    continue

                return LatestTimestampDirPath, LatestLeafFolderPath, ModelFiles

        return None

    def GetValidLatestTimestampDirInfo(self) :
        
        return self.TravelValidLatestTimestampDirInfo(lambda files : files is None)

    def GetValidLatestTimestampDirPath(self) :
        LatestTimestampDirPath, _, _ = self.GetValidLatestTimestampDirInfo()
        if self.__RootPath is None :
            self.__RootPath = LatestTimestampDirPath
        return LatestTimestampDirPath
        
    def GetFileFromValidLatestTimestampDirPath(self, **inKWArgs) :
        ValidPath = self.GetValidLatestTimestampDirPath()
        if ValidPath is None:
            return None, None
        
        LeafDirName = self.MakeLeafDirName(**inKWArgs)
        if LeafDirName is None:
            return None, None

        LeafDirFullPath = os.path.join(ValidPath, LeafDirName)

        FileName = self.MakeFileName(**inKWArgs)
        if FileName is None:
            return None, None
        
        if FileName.endswith(self.Extension) == False:
            FileName = FileName + self.Extension
        
        return LeafDirFullPath, FileName
    
    
    def GetFilePathAndNameFromTimestampDirPathByEpoch(self, **inKWArgs) :
        LeafDirName = self.MakeLeafDirName(**inKWArgs)
        if LeafDirName is None:
            return None, None
        
        FileName = self.MakeFileName(**inKWArgs)
        if FileName is None:
            return None, None
        
        if FileName.endswith(self.Extension) == False:
            FileName = FileName + self.Extension
        
        AllTimestampDirNames = self.GetAllTimestampDirNames()
        if len(AllTimestampDirNames) == 0:
            return None, None
        
        AllTimestampDirNames.sort(key=lambda x: int(x), reverse=True)

        for DirName in AllTimestampDirNames:
            ModelFilePath = os.path.join(self.RawRootPath, DirName, LeafDirName)
            ModelFile = os.path.join(ModelFilePath, FileName)

            if os.path.exists(ModelFile):
                return ModelFilePath, FileName

        return None, None
