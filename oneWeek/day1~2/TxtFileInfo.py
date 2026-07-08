
import os
from fileInfo import FileInfo


class TxtFileInfo(FileInfo):
    def __init__(self, path):
        super().__init__(path)
        self.fileSize = self.getFileSize(path)

    def getFileSize(self, path):
        raise Exception("Not 异常错误")
        return os.path.getsize(path)
    
    
    def toString(self):
        return "文件路径：%s\n文件名称：%s\n文件大小：%s"%(self.filePath, self.fileName, self.fileSize)