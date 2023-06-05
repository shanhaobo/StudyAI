import os

#root directory->....->Leaf directory(terminal directory)
class BaseFileManager() :
    def __init__(self, inRootDir:str, inExtension:str) -> None:
        self.RootDir = inRootDir
        self.Extension = inExtension if inExtension.startswith(".") else ("." + inExtension)

    def MakeLeafDirName(self, **inKWArgs) -> str:
        pass

    def MakeFileName(self, **inKWArgs) -> str:
        pass

    def MakeFileFullPath(self, **inKWArgs):
        LeafDirName = self.MakeLeafDirName(**inKWArgs)
        if not LeafDirName:
            return None

        LeafDirFullPath = os.path.join(self.RootDir, LeafDirName)
        os.makedirs(LeafDirFullPath, exist_ok=True)

        FileName = self.MakeFileName(**inKWArgs)
        if not FileName:
            return None
        
        if FileName.endswith(self.Extension) == False:
            FileName = FileName + self.Extension
        
        return os.path.join(LeafDirFullPath, FileName)

    def GetAllLeafDirNames(self) :
        pass
