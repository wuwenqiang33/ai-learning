"""
Step 08 — 混合检索（Ensemble Retrieval）（Day 13-14 核心成果）

功能：结合向量检索（语义理解）和 BM25 检索（关键词匹配）的优势。
向量检索擅长理解语义，但容易忽略精确关键词；
BM25 擅长精确匹配，但对语义泛化能力弱。
混合检索 = 向量 Top-K + BM25 Top-K → 合并去重 → 重排序

运行前准备：
    pip install rank-bm25

运行: python steps/step08_ensemble.py
"""

import sys, os

from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.append(str(Path(__file__).parent.parent))

import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
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
print("Step 08 — 混合检索（向量 + BM25）")
print("=" * 60)

# ── 1. 加载文档 ──
print("\n[1] 加载文档...")
DOC_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "kkfile.md")
docs = TextLoader(DOC_PATH, encoding="utf-8").load()
print(f"    加载了 {len(docs)} 篇文档")

# ── 2. 文本分割 ──
print("[2] 文本分割...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"    分成 {len(chunks)} 个块")

# ── 3. 构建两种检索器 ──
print("[3] 构建检索器...")

# 3a. 向量检索器（PgVector）
print("    [3a] 构建 PgVector 向量检索器...")
embeddings = BigModelEmbeddingFunc(api_key=EMBEDDING_KEY)

from langchain_postgres.vectorstores import PGVector as PGVectorStore

vectorstore = PGVectorStore(
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
    connection=PG_CONNECTION,
    use_jsonb=True,
)

count = 0
try:
    coll_info = vectorstore.get_collection(COLLECTION_NAME)
    count = coll_info.get("item_count", 0) if coll_info else 0
except Exception:
    count = 0

if count == 0:
    vectorstore.add_documents(chunks)
    print(f"    新建 PgVector 集合，存入 {len(chunks)} 个分块")
else:
    print(f"    加载已有 PgVector 集合，共 {count} 条记录")

vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10},
)

# 3b. BM25 检索器
print("    [3b] 构建 BM25 关键词检索器...")
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10

print("    两种检索器已就绪")

# ── 4. 混合检索函数 ──
def ensemble_retrieve(query: str, top_k: int = 5) -> list:
    """
    混合检索：向量检索 + BM25 检索，合并去重后返回 Top-K。

    策略：
    1. 向量检索 Top-K（语义理解）
    2. BM25 检索 Top-K（关键词匹配）
    3. 合并去重，按向量检索排名优先
    4. 返回 Top-K

    Java 类比：就像同时查 Elasticsearch 的两个索引（全文 + 向量），
    然后把结果合并，去重后取前 K 个。
    """
    # 并行检索
    vector_results = vector_retriever.invoke(query)
    bm25_results = bm25_retriever.invoke(query)

    print(f"\n    向量检索返回: {len(vector_results)} 条")
    print(f"    BM25 检索返回: {len(bm25_results)} 条")

    # 合并去重（按 page_content 哈希）
    seen = set()
    merged = []

    # 优先保留向量检索的结果（语义理解更重要）
    for doc in vector_results:
        key = hash(doc.page_content)
        if key not in seen:
            seen.add(key)
            merged.append((doc, 0))  # 0 = 向量检索优先

    for doc in bm25_results:
        key = hash(doc.page_content)
        if key not in seen:
            seen.add(key)
            merged.append((doc, 1))  # 1 = BM25 补充

    # 取前 top_k 个
    top_docs = [doc for doc, _ in merged[:top_k]]

    print(f"    合并去重后: {len(top_docs)} 条")
    return top_docs


# ── 5. 对比实验 ──
print("\n[4] 对比实验...")
test_queries = [
    "kkFileView支持哪些文档格式？",
    "v5.0.0版本有什么新功能？",
    "官网地址是什么？",
]

for query in test_queries:
    print(f"\n  查询: {query}")

    # 单独向量检索
    vector_only = vector_retriever.invoke(query)
    print(f"    仅向量检索: {len(vector_only)} 条")

    # 单独 BM25 检索
    bm25_only = bm25_retriever.invoke(query)
    print(f"    仅 BM25 检索: {len(bm25_only)} 条")

    # 混合检索
    ensemble = ensemble_retrieve(query, top_k=3)
    print(f"    混合检索: {len(ensemble)} 条")

    # 展示第一条结果
    if ensemble:
        print(f"    首条内容: {ensemble[0].page_content[:80]}...")

# ── 6. 构建混合检索 RAG 管道 ──
print("\n[5] 构建混合检索 RAG 管道...")

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

# 用混合检索器替换原来的单一检索器
def ensemble_step(x):
    """混合检索步骤，接收 dict {"input": query_str}"""
    query = x["input"]
    docs = ensemble_retrieve(query, top_k=5)
    return {"context": "\n\n".join(d.page_content for d in docs), "input": query}

rag_chain_ensemble = (
    RunnableLambda(lambda x: {"input": x})
    | ensemble_step
    | prompt
    | llm
    | StrOutputParser()
)

# ── 7. 问答测试 ──
print("\n[6] 混合检索 RAG 问答测试...")
for q in test_queries:
    print(f"\n  Q: {q}")
    result = rag_chain_ensemble.invoke(q)
    print(f"  A: {result[:200]}...")

print("\n" + "=" * 60)
print("Step 08 完成 — 下一步: step09_query_rewrite.py（查询改写）")
print("=" * 60)
