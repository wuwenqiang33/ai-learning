"""
Step 01 — 最简 RAG 系统（Day 1-2 基础概念）

功能：读取一个 Markdown 文件，分割成块，生成向量存入内存向量库，
      然后回答用户问题。

这是 RAG 的"Hello World"——一切从这里开始。

运行: python steps/step01_basic.py
"""

import sys, os

# 将 lib/ 加入路径
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.append(str(Path(__file__).parent.parent))
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from lib.embedding import BigModelEmbeddingFunc
from lib.config import load_config

# ── 加载配置 ──
config = load_config()
EMBEDDING_KEY = config["embedding"]["api_key"]
DOC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "docs", "kkfile.md"
)

print("=" * 50)
print("Step 01 — 最简 RAG 系统")
print("=" * 50)

# 1. 加载文档
print("\n[1] 加载文档...")
docs = TextLoader(DOC_PATH, encoding="utf-8").load()
print(f"    加载了 {len(docs)} 篇文档")

# 2. 文本分割
print("[2] 文本分割...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"    分成 {len(chunks)} 个块，平均长度 ~{sum(len(c.page_content) for c in chunks) // len(chunks)} 字符")

# 3. 向量化 + 存储（内存中）
print("[3] 向量化 + 存储...")
embeddings = BigModelEmbeddingFunc(api_key=EMBEDDING_KEY)
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)
print("    向量库已就绪")

# 4. 问答测试
print("[4] 问答测试...")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template(
    "请严格依据下面参考资料回答用户问题，给出回答的依据文件，不要编造内容：\n"
    "参考资料：{context}\n"
    "用户问题：{input}"
)

llm = ChatOpenAI(
    model=config["deepseek"]["model"],
    temperature=0.7,
    api_key=config["deepseek"]["api_key"],
    base_url=config["deepseek"]["base_url"],
)

rag_chain = {"context": retriever, "input": RunnablePassthrough()} | prompt | llm | StrOutputParser()

questions = ["kkFileView支持哪些文档格式？", "官网地址是什么？"]
for q in questions:
    print(f"\n  Q: {q}")
    result = rag_chain.invoke(q)
    print(f"  A: {result[:150]}...")

print("\n" + "=" * 50)
print("Step 01 完成 — 下一步: step03_full_rag.py（持久化向量库）")
print("=" * 50)
