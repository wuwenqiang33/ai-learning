# 多格式文档加载器 - 使用说明

## 概述

本模块实现了基于 LangChain 的统一文档加载器，支持多种常见文件格式的解析。

## 快速开始

```python
from multi_format_loader import MultiFormatDocumentLoader

loader = MultiFormatDocumentLoader(encoding="utf-8")

# 加载单个文件
docs = loader.load("document.pdf")

# 批量加载整个文件夹
all_docs = loader.load_batch("./documents/")
```

## 支持的格式

| 扩展名 | 加载器 | 说明 |
|--------|--------|------|
| .pdf | PyPDFLoader | 基于 pypdf 引擎 |
| .docx | Docx2txtLoader | 提取 Word 文本 |
| .xlsx | UnstructuredExcelLoader | 解析 Excel 表格 |
| .csv | UnstructuredExcelLoader | CSV 作为电子表格解析 |
| .pptx | UnstructuredPowerPointLoader | 提取投影片内容 |
| .md | TextLoader | Markdown 纯文本 |
| .txt | TextLoader | UTF-8 编码文本 |

## 代码示例

以下是加载器核心代码的结构：

```python
class MultiFormatDocumentLoader:
    LOADERS = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.md': TextLoader,
        '.txt': TextLoader,
        '.csv': UnstructuredExcelLoader,
    }

    def load(self, file_path: str) -> list:
        ext = Path(file_path).suffix.lower()
        loader_class = self.LOADERS.get(ext)
        loader = loader_class(file_path)
        return loader.load()
```

## 注意事项

1. 确保安装了所有依赖包
2. PDF 文件建议为文本型（非扫描图片）
3. Excel 文件会自动解析第一个工作表
4. 中文内容请确保文件编码为 UTF-8

## 参考链接

- [LangChain 文档加载器](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [Unstructured 库文档](https://docs.unstructured.io/)
