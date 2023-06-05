import os

#root directory->....->Leaf directory(terminal directory)
class BaseFileManager() :
    def __init__(self, inRootPath:str, inExtension:str) -> None:
        self.RootPath = inRootPath
        self.Extension = inExtension if inExtension.startswith(".") else ("." + inExtension)
        os.makedirs(inRootPath, exist_ok=True)

    def MakeLeafDirName(self, **inKWArgs) -> str:
        pass

    def MakeFileName(self, **inKWArgs) -> str:
        pass

    def MakeFileFullPath(self, **inKWArgs):
        LeafDirName = self.MakeLeafDirName(**inKWArgs)
        if not LeafDirName:
            return None

        LeafDirFullPath = os.path.join(self.RootPath, LeafDirName)
        os.makedirs(LeafDirFullPath, exist_ok=True)

        FileName = self.MakeFileName(**inKWArgs)
        if not FileName:
            return None
        
        if FileName.endswith(self.Extension) == False:
            FileName = FileName + self.Extension
        
        return os.path.join(LeafDirFullPath, FileName)

    def GetAllSubDirNames(self, inPath = None) :
        if not inPath :
            inPath = self.RootPath

        return [d for d in os.listdir(inPath) if (os.path.isdir(os.path.join(inPath, d)))]
    
    def GetAllSubDirPaths(self, inPath = None) :
        if not inPath :
            inPath = self.RootPath

        return [p for d in os.listdir(inPath) if (os.path.isdir(p := os.path.join(inPath, d)))]

    
    

