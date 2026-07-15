"""
Step 02 — ChromaDB 基础 CRUD（Day 3-4 向量数据库入门）

功能：学习 ChromaDB 的基本操作 —— 增删改查、语义搜索、元数据过滤。
这是理解向量数据库的必经之路。

运行: python steps/step02_chroma.py
"""

import sys, os

from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.append(str(Path(__file__).parent.parent))

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from lib.config import load_config

config = load_config()
EMBEDDING_KEY = config["embedding"]["api_key"]
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "chroma_step02")

print("=" * 50)
print("Step 02 — ChromaDB 基础 CRUD")
print("=" * 50)


# ── 自定义 Embedding 函数（适配 ChromaDB 原生 API） ──
class BigModelEmbedding(EmbeddingFunction):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = config["embedding"]["base_url"]

    def __call__(self, texts: Documents) -> Embeddings:
        import requests

        resp = requests.post(
            url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": "embedding-3", "input": texts},
        )
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]


# ── 1. 创建客户端和集合 ──
print("\n[1] 创建 ChromaDB 客户端...")
client = chromadb.PersistentClient(path=DB_PATH)
coll = client.get_or_create_collection(
    name="demo",
    embedding_function=BigModelEmbedding(EMBEDDING_KEY),
    metadata={"hnsw:space": "cosine"},
)
print(f"    集合: demo, 文档数: {coll.count()}")

# ── 2. 写入数据（INSERT） ──
print("\n[2] 写入数据...")
coll.add(
    documents=[
        "Chroma 是轻量级向量数据库",
        "BigModel API 可以生成文本 Embedding 向量",
        "RAG 架构依靠向量库检索知识库",
    ],
    metadatas=[{"source": "note1"}, {"source": "note2"}, {"source": "note3"}],
    ids=["id1", "id2", "id3"],
)
print(f"    写入 3 条，当前总数: {coll.count()}")

# ── 3. 更新数据（UPDATE） ──
print("\n[3] 更新数据...")
coll.update(
    ids=["id1"],
    documents=["Chroma 是轻量级向量数据库，支持多种存储后端"],
    metadatas=[{"source": "note1"}],
)
print("    更新了 id1")

# ── 4. 删除数据（DELETE） ──
print("\n[4] 删除数据...")
coll.delete(ids=["id3"])
print(f"    删除 id3，当前总数: {coll.count()}")

# ── 5. 语义查询 ──
print("\n[5] 语义查询...")
results = coll.query(query_texts=["向量数据库能用来做什么"], n_results=2)
print(f"    查询: '向量数据库能用来做什么'")
for i, doc in enumerate(results["documents"][0]):
    dist = results["distances"][0][i]
    print(f"    [{i+1}] (距离={dist:.4f}) {doc}")

# ── 6. 元数据过滤 ──
print("\n[6] 元数据过滤查询...")
results = coll.query(
    query_texts=["向量数据库"],
    where={"source": "note1"},
    n_results=2,
)
print(f"    查询: source='note1' 的文档")
for doc in results["documents"][0]:
    print(f"    - {doc}")

print("\n" + "=" * 50)
print("Step 02 完成 — 下一步: step03_full_rag.py（端到端 RAG）")
print("=" * 50)
