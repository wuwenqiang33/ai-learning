"""
Step 05 — Rerank 重排序（Day 9 进阶）

功能：在 RAG 管道中加入 Rerank 步骤：
      向量检索召回 Top-K → Rerank 精排 → 取 Top-N 喂给 LLM

原理：
  - 向量检索 (Embedding): 快但粗糙，用余弦相似度粗筛候选
  - Rerank (LLM 打分): 慢但精准，逐对计算 query-doc 相关性分数
  - 组合效果 ≈ 召回率 × 排序精度 的双重保障

运行: python steps/step05_rerank.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.append(str(Path(__file__).parent.parent))

from lib.embedding import BigModelEmbeddingFunc
from lib.config import load_config

config = load_config()
EMBEDDING_KEY = config["embedding"]["api_key"]
DEEPSEEK_KEY = config["deepseek"]["api_key"]
DEEPSEEK_URL = config["deepseek"]["base_url"]
DEEPSEEK_MODEL = config["deepseek"]["model"]
CHROMA_DB_PATH = config["chroma"]["persist_directory"]
DOC_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "kkfile.md")


def simple_rerank(
    query: str, docs: list[Document], top_k: int = 2
) -> list[Document]:
    """
    简易 Rerank：用 LLM 对每个文档打分，取最高分。
    生产环境建议用专门的 Rerank 模型（如 BGE-Reranker、Cohere Rerank）。
    """
    if len(docs) <= top_k:
        return docs

    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL, temperature=0,
        api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_URL,
    )

    rerank_prompt = PromptTemplate.from_template(
        "你是一个文档相关性评分器。请对以下每份文档与查询的相关性打分（1-5分）。\n"
        "查询: {query}\n\n"
        "文档列表:\n{doc_list}\n\n"
        "请按以下 JSON 格式返回，只返回 JSON，不要其他内容:\n"
        '[{{"index": 0, "score": 5}}, {{"index": 1, "score": 3}}]'
    )

    doc_list = "\n---\n".join(
        f"[文档{i}]\n{d.page_content[:300]}" for i, d in enumerate(docs)
    )

    response = llm.invoke(rerank_prompt.format(query=query, doc_list=doc_list))
    import re

    json_match = re.search(r"\[.*\]", response.content, re.DOTALL)
    if json_match:
        scores = json.loads(json_match.group())
        scores.sort(key=lambda x: x["score"], reverse=True)
        top_indices = [s["index"] for s in scores[:top_k]]
        return [docs[i] for i in top_indices if i < len(docs)]

    return docs[:top_k]


print("=" * 60)
print("Step 05 — Rerank 重排序实验")
print("=" * 60)

# 加载文档 + 向量库
docs = TextLoader(DOC_PATH, encoding="utf-8").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = BigModelEmbeddingFunc(api_key=EMBEDDING_KEY)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)

query = "kkFileView支持哪些3D模型格式？"

# 无 Rerank: Top-3
print(f"\n查询: {query}")
print(f"\n--- 无 Rerank (Top-3) ---")
retriever_no = vectorstore.as_retriever(search_kwargs={"k": 3})
docs_no = retriever_no.invoke(query)
for i, d in enumerate(docs_no):
    print(f"  [{i+1}] {d.page_content[:100]}...")

# 有 Rerank: Top-10 → Top-2
print(f"\n--- 有 Rerank (Top-10 → Top-2) ---")
retriever_yes = vectorstore.as_retriever(search_kwargs={"k": 10})
docs_yes = retriever_yes.invoke(query)
reranked = simple_rerank(query, docs_yes, top_k=2)
for i, d in enumerate(reranked):
    print(f"  [{i+1}] {d.page_content[:100]}...")

print("\n结论: Rerank 过滤掉了不相关的候选，让 LLM 只看最相关的 2 条")
print("\n" + "=" * 60)
print("Step 05 完成 — 下一步: step06_multiformat.py（多格式最终版）")
print("=" * 60)
