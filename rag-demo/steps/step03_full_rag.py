"""
Step 03 — 端到端 RAG 文档问答系统（Day 5-7 核心成果）

功能：完整的 RAG 流水线 —— 加载文档 → 分割 → 向量化 → 持久化到 ChromaDB
      → 构建检索问答链 → 交互式提问。

这是第一篇博客的核心成果，也是后续所有升级的基础。

运行: python steps/step03_full_rag.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from embedding import BigModelEmbeddingFunc
from config import load_config

# ── 加载配置 ──
config = load_config()
EMBEDDING_KEY = config["embedding"]["api_key"]
DEEPSEEK_KEY = config["deepseek"]["api_key"]
DEEPSEEK_URL = config["deepseek"]["base_url"]
DEEPSEEK_MODEL = config["deepseek"]["model"]
CHROMA_DB_PATH = config["chroma"]["persist_directory"]
DOC_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "kkfile.md")

print("=" * 50)
print("Step 03 — 端到端 RAG 文档问答系统")
print("=" * 50)

# ── 1. 加载文档 ──
print("\n[1] 加载文档...")
docs = TextLoader(DOC_PATH, encoding="utf-8").load()
print(f"    加载了 {len(docs)} 篇文档")

# ── 2. 文本分割 ──
print("[2] 文本分割 (chunk_size=500, overlap=50)...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"    分成 {len(chunks)} 个块")

# ── 3. 向量化 + 持久化 ──
print("[3] 向量化 + 存储...")
embeddings = BigModelEmbeddingFunc(api_key=EMBEDDING_KEY)

if os.path.exists(CHROMA_DB_PATH):
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )
    print(f"    加载已有向量库，共 {vectorstore._collection.count()} 条记录")
else:
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_DB_PATH
    )
    vectorstore.persist()
    print(f"    新建向量库，存入 {len(chunks)} 个分块")

# ── 4. 构建 RAG 管道 ──
print("[4] 构建 RAG 管道...")

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
questions = ["kkFileView支持哪些文档格式？", "官网地址是什么？"]
for q in questions:
    print(f"\n  Q: {q}")
    result = rag_chain.invoke(q)
    print(f"  A: {result[:200]}...")

print("\n" + "=" * 50)
print("Step 03 完成 — 下一步: step04_chunk_exp.py（分割策略实验）")
print("=" * 50)
