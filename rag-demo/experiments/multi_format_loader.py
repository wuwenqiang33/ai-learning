# 多格式文档解析器

"""
支持 PDF/Word/Excel/PPT/Markdown 的统一文档解析器
类比 Java 的策略模式：不同格式 = 不同策略，通过扩展名自动选择
"""

import os
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)


class MultiFormatDocumentLoader:
    """统一文档解析器"""

    # 格式 -> Loader 映射表
    LOADERS = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.md': TextLoader,
        '.txt': TextLoader,
        '.csv': TextLoader,
    }

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def load(self, file_path: str) -> list:
        """加载单个文件，返回 Document 列表"""
        ext = Path(file_path).suffix.lower()
        loader_class = self.LOADERS.get(ext)

        if not loader_class:
            raise ValueError(f"不支持的文件格式: {ext}")

        # 实例化对应的 Loader
        if ext in ('.pdf', '.xlsx','.docx','.pptx'):
            loader = loader_class(file_path)
        else:
            loader = loader_class(file_path, encoding=self.encoding)

        docs = loader.load()

        # 统一添加元数据
        for doc in docs:
            doc.metadata.update({
                "source_file": file_path,
                "file_type": ext,
                "file_name": Path(file_path).name,
                "load_time": datetime.now().isoformat(),
            })

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


if __name__ == "__main__":
    loader = MultiFormatDocumentLoader()

    # 测试md单文件加载
    # doc_path = os.path.join(os.path.dirname(__file__), "doc", "kkfile.md")
    # if os.path.exists(doc_path):
    #     docs = loader.load(doc_path)
    #     print(f"加载了 {len(docs)} 页")
    #     print(f"第一页内容预览: {docs[0].page_content[:200]}")
    #     print(f"元数据: {docs[0].metadata}")

    # 测试pdf单文件加载
    # doc_path = os.path.join(os.path.dirname(__file__), "doc", "阿里巴巴Java开发手册.pdf")
    # if os.path.exists(doc_path):
    #     docs = loader.load(doc_path)
    #     print(f"加载了 {len(docs)} 页")
    #     print(f"第10页内容预览: {docs[9].page_content[:200]}")
    #     print(f"元数据: {docs[0].metadata}")
    
    # 测试csv单文件加载
    # doc_path = os.path.join(os.path.dirname(__file__), "doc", "test.csv")
    # if os.path.exists(doc_path):
    #     docs = loader.load(doc_path)
    #     print(f"加载了 {len(docs)} 页")
    #     print(f"第1页内容预览: {docs[0].page_content[:200]}")
    #     print(f"元数据: {docs[0].metadata}")

    # 测试docx单文件加载
    # doc_path = os.path.join(os.path.dirname(__file__), "doc", "test.docx")
    # if os.path.exists(doc_path):
    #     docs = loader.load(doc_path)
    #     print(f"加载了 {len(docs)} 页")
    #     print(f"第1页内容预览: {docs[0].page_content[:200]}")
    #     print(f"元数据: {docs[0].metadata}")

    # 测试xlsx单文件加载
    # doc_path = os.path.join(os.path.dirname(__file__), "doc", "test.xlsx")
    # if os.path.exists(doc_path):
    #     docs = loader.load(doc_path)
    #     print(f"加载了 {len(docs)} 页")
    #     print(f"第1页内容预览: {docs[0].page_content[:200]}")
    #     print(f"元数据: {docs[0].metadata}")
    
    # 测试pptx单文件加载
    # doc_path = os.path.join(os.path.dirname(__file__), "doc", "test.pptx")
    # if os.path.exists(doc_path):
    #     docs = loader.load(doc_path)
    #     print(f"加载了 {len(docs)} 页")
    #     print(f"第1页内容预览: {docs[0].page_content[:200]}")
    #     print(f"元数据: {docs[0].metadata}")

    # 测试批量加载
    docs = loader.load_batch(os.path.join(os.path.dirname(__file__), "doc"))
    print(f"加载了 {len(docs)} 个文档块")
    
