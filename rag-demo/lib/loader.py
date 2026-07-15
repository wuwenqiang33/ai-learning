"""
统一文档解析器 — 支持 PDF/Word/Excel/PPT/Markdown/Text/CSV

设计思路：类似 Java 的策略模式，不同格式 = 不同策略，通过扩展名自动选择。
统一输出 LangChain 的 Document 对象，并自动注入元数据。

使用方式:
    from loader import MultiFormatDocumentLoader
    loader = MultiFormatDocumentLoader()
    docs = loader.load("doc/kkfile.pdf")
    all_docs = loader.load_batch("doc/")
"""

import os
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

# 按需导入（避免未安装时报错）
try:
    from langchain_community.document_loaders import UnstructuredExcelLoader
except ImportError:
    UnstructuredExcelLoader = None

try:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
except ImportError:
    UnstructuredPowerPointLoader = None


class MultiFormatDocumentLoader:
    """统一文档解析器"""

    # 格式 -> Loader 映射表
    LOADERS = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".md": TextLoader,
        ".txt": TextLoader,
    }

    # 可选格式（需要额外 pip install）
    OPTIONAL_LOADERS = {
        ".xlsx": UnstructuredExcelLoader,
        ".csv": UnstructuredExcelLoader,
        ".pptx": UnstructuredPowerPointLoader,
    }

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding
        # 合并可选 Loader（None 的跳过）
        for ext, cls in self.OPTIONAL_LOADERS.items():
            if cls is not None:
                self.LOADERS[ext] = cls

    def load(self, file_path: str) -> list:
        """加载单个文件，返回 Document 列表"""
        ext = Path(file_path).suffix.lower()
        loader_class = self.LOADERS.get(ext)

        if not loader_class:
            raise ValueError(f"不支持的文件格式: {ext}")

        # 实例化对应的 Loader
        if ext in (".pdf", ".xlsx", ".docx", ".pptx"):
            loader = loader_class(file_path)
        else:
            loader = loader_class(file_path, encoding=self.encoding)

        docs = loader.load()

        # 统一添加元数据
        for doc in docs:
            doc.metadata.update(
                {
                    "source_file": file_path,
                    "file_type": ext,
                    "file_name": Path(file_path).name,
                    "load_time": datetime.now().isoformat(),
                }
            )

        return docs

    def load_batch(self, folder_path: str) -> list:
        """批量加载文件夹下所有支持的文档"""
        all_docs = []
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")

        for file_path in folder.glob("*"):
            if file_path.suffix.lower() in self.LOADERS:
                try:
                    docs = self.load(str(file_path))
                    all_docs.extend(docs)
                    print(f"  ✅ {file_path.name}: {len(docs)} 页")
                except Exception as e:
                    print(f"  ❌ {file_path.name}: {e}")

        print(f"\n  共加载 {len(all_docs)} 个文档块")
        return all_docs
