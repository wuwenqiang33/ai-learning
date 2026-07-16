"""
Step 06 — 多格式 RAG 系统（Day 10 简单完整版）

功能：支持 PDF/Word/Markdown/Text 多格式文档的统一 RAG 系统。
整合了 MultiFormatDocumentLoader + 智能分割 + 元数据管理。

这是整个学习过程的最终成果，可以直接作为项目模板使用。

运行: python steps/step06_multiformat.py
"""

import sys, os

from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from lib.embedding import BigModelEmbeddingFunc
from lib.loader import MultiFormatDocumentLoader
from lib.config import load_config

# ── 加载配置 ──
config = load_config()
DEEPSEEK_KEY = config["deepseek"]["api_key"]
DEEPSEEK_URL = config["deepseek"]["base_url"]
DEEPSEEK_MODEL = config["deepseek"]["model"]
EMBEDDING_KEY = config["embedding"]["api_key"]
CHROMA_DB_PATH = config["chroma"]["persist_directory"]

print("=" * 60)
print("Step 06 — 多格式 RAG 系统（最终完整版）")
print("=" * 60)

# ── 1. 加载多格式文档 ──
print("\n[1] 加载多格式文档...")
loader = MultiFormatDocumentLoader()

doc_files = [
    "docs/国务院关于《扩大消费“十五五”规划》的批复_国务院文件_中国政府网.pdf",
]

all_docs = []
for doc_file in doc_files:
    full_path = os.path.join(os.path.dirname(__file__), "..", doc_file)
    if os.path.exists(full_path):
        docs = loader.load(full_path)
        all_docs.extend(docs)
        print(f"    ✅ {doc_file}: {len(docs)} 页")
    else:
        print(f"    ⚠️  文件不存在: {full_path}")

print(f"    总计: {len(all_docs)} 个文档块")

# ── 2. 文本分割（根据格式选择策略） ──
print("\n[2] 文本分割...")

markdown_chunks = []
other_chunks = []

for doc in all_docs:
    if doc.metadata.get("file_type") == ".md":
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "title")]
        )
        markdown_chunks.extend(header_splitter.split_text(doc.page_content))
    else:
        other_chunks.append(doc)

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50
)
other_split = recursive_splitter.split_documents(other_chunks)

all_chunks = markdown_chunks + other_split
print(f"    Markdown 分割: {len(markdown_chunks)} 块")
print(f"    其他格式分割: {len(other_split)} 块")
print(f"    总计: {len(all_chunks)} 个分块")

# ── 3. 向量化 + 存储 ──
print("\n[3] 向量化 + 存储...")
embeddings = BigModelEmbeddingFunc(api_key=EMBEDDING_KEY)

if os.path.exists(CHROMA_DB_PATH):
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )
    print(f"    加载已有向量库，共 {vectorstore._collection.count()} 条记录")
else:
    vectorstore = Chroma.from_documents(
        all_chunks, embeddings, persist_directory=CHROMA_DB_PATH
    )
    vectorstore.persist()
    print(f"    新建向量库，存入 {len(all_chunks)} 个分块")

# ── 4. 构建 RAG 管道 ──
print("\n[4] 构建 RAG 管道...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt = PromptTemplate.from_template(
    "请严格依据下面参考资料回答用户问题，给出回答的依据文件，不要编造内容：\n"
    "参考资料：{context}\n"
    "用户问题：{input}"
)

llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    temperature=0.7,
    api_key=DEEPSEEK_KEY,
    base_url=DEEPSEEK_URL,
)

rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 5. 问答测试 ──
print("\n[5] 问答测试...")
questions = [
    "扩大消费“十五五”规划的总体要求有哪些？",
    "大力优化消费环境是指什么？",
]
    

for question in questions:
    print(f"\n  Q: {question}")
    result = rag_chain.invoke(question)
    print(f"  A: {result[:200]}...")

print("\n" + "=" * 60)
print("Step 06 完成 — 这就是 RAG 系统！")
print("=" * 60)
