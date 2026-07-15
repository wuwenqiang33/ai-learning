"""
Step 07 — PgVector 完整 RAG 系统（Day 11-12 核心成果）

功能：从 ChromaDB 切换到 PgVector，更接近生产环境的向量数据库方案。
PgVector 是 PostgreSQL 的扩展，让你可以在熟悉的 SQL 环境中做向量检索。

运行前准备：
    1. 安装 PostgreSQL
    2. CREATE EXTENSION vector;
    3. pip install psycopg2-binary langchain-postgres
    4. 确保 config.json 中有 chroma.persist_directory（兼容旧步骤）

运行: python steps/step07_pgvector.py
"""

import sys, os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.append(str(Path(__file__).parent.parent))

import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from lib.embedding import BigModelEmbeddingFunc
from lib.config import load_config

# ── 加载配置 ──
config = load_config()
EMBEDDING_KEY = config["embedding"]["api_key"]
DEEPSEEK_KEY = config["deepseek"]["api_key"]
DEEPSEEK_URL = config["deepseek"]["base_url"]
DEEPSEEK_MODEL = config["deepseek"]["model"]

# PgVector 连接信息
PG_CONNECTION = (
    "postgresql://root:rootPassword@localhost:5432/rag_db"
)
COLLECTION_NAME = "kkfile_documents"

print("=" * 60)
print("Step 07 — PgVector 完整 RAG 系统")
print("=" * 60)

# ── 1. 加载文档 ──
print("\n[1] 加载文档...")
DOC_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "kkfile.md")
docs = TextLoader(DOC_PATH, encoding="utf-8").load()
print(f"    加载了 {len(docs)} 篇文档")

# ── 2. 文本分割 ──
print("[2] 文本分割 (chunk_size=500, overlap=50)...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"    分成 {len(chunks)} 个块")

# ── 3. 向量化 + PgVector 存储 ──
print("[3] 向量化 + 存储到 PgVector...")
embeddings = BigModelEmbeddingFunc(api_key=EMBEDDING_KEY)

from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector as PGVectorStore

# 清空旧数据（首次运行可注释掉）
# PGVectorStore.drop_tables(engine=None)

vectorstore = PGVectorStore(
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
    connection=PG_CONNECTION,
    use_jsonb=True,
)

# 检查是否已有数据
try:
    coll_info = vectorstore.get_collection(COLLECTION_NAME)
    count = coll_info.get("item_count", 0) if coll_info else 0
except Exception:
    count = 0

if count > 0:
    print(f"    加载已有 PgVector 集合，共 {count} 条记录")
else:
    vectorstore.add_documents(chunks)
    print(f"    新建 PgVector 集合，存入 {len(chunks)} 个分块")

# ── 4. 构建 RAG 管道 ──
print("[4] 构建 RAG 管道...")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

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
    "kkFileView支持哪些文档格式？",
    "kkFileView的官网地址是什么？",
    "v5.0.0版本有什么新功能？",
]
for q in questions:
    print(f"\n  Q: {q}")
    result = rag_chain.invoke(q)
    print(f"  A: {result[:200]}...")

# ── 6. 元数据过滤查询（PgVector 特色功能） ──
print("\n[6] 元数据过滤查询...")
retriever_filtered = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "filter": {"source_file": "docs/kkfile.md"},
    },
)

filtered_result = retriever_filtered.invoke("kkFileView支持哪些文档格式？")
print(f"    过滤后检索到 {len(filtered_result)} 条相关文档")
for i, doc in enumerate(filtered_result[:2]):
    print(f"    [{i+1}] 来源: {doc.metadata.get('source_file', 'N/A')}")
    print(f"        内容: {doc.page_content[:80]}...")

print("\n" + "=" * 60)
print("Step 07 完成 ")
print("=" * 60)
