# 文件信息类
import os
class FileInfo(object):
    def __init__(self, path):
        self.path = path
        self.fileName = self.getFileName(path)
        self.filePath = FileInfo.getFilePath(path)

    def getFileName(self, path):
        return os.path.basename(path)
    @staticmethod
    def getFilePath(path):
        
        return os.path.dirname(path)
    
    def toString(self):
        return self.fileName + " " + self.filePath
