# 分割策略对比

from multi_format_loader import MultiFormatDocumentLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
import os



if __name__ == "__main__":
    loader = MultiFormatDocumentLoader()
    doc_path = os.path.join(os.path.dirname(__file__), "doc", "kkfile.md")
    docs = loader.load(doc_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 每块最多 500 字符
        chunk_overlap=50,    # 相邻块重叠 50 字符
        separators=[
            "\n\n",          # 优先按段落分割
            "\n",            # 其次按行分割
            " ",             # 然后按词分割
            ""               # 最后强制截断
        ]
    )
    chunks = splitter.split_documents(docs)

    # 打印第1页
    # print("第一页：",chunks[0].page_content)
    # print ("type:",chunks[0].type)
    # 打印前三页的内容
    for index in range(0 , 3):
        print(index,chunks[index].page_content,sep = ':')
    # for index , chunk in enumerate(chunks):
    #     print(index,chunks,sep = ':')

    headers_to_split_on = [
    ("#", "一级标题"),
    ("##", "二级标题"),
    ("###", "三级标题"),
]
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = splitter.split_text(docs[0].page_content)

    # 每个 chunk 的 metadata 会携带标题信息
    for chunk in chunks[:3]:
        print(f"标题: {chunk.metadata}")
        print(f"内容: {chunk.page_content[:100]}...")






