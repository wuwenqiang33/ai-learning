# Rerank 重排序集成示例

"""
在 RAG 管道中加入 Rerank 步骤：
  向量检索召回 Top-K → Rerank 精排 → 取 Top-N 喂给 LLM

原理:
  - 向量检索 (Embedding): 快但粗糙，用余弦相似度粗筛候选
  - Rerank (Cross-Encoder): 慢但精准，逐对计算 query-doc 相关性分数
  - 组合效果 ≈ 召回率 × 排序精度 的双重保障
"""

import os
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from BigModelEmbedding import BigModelEmbeddingFunc
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ========== 配置 ==========
with open(os.path.join(os.path.dirname(__file__), "config_bigmodel.json"), "r", encoding="utf-8") as f:
    config = json.load(f)

EMBEDDING_KEY = config["embedding"]["api_key"]
DEEPSEEK_KEY = config["deepseek"]["api_key"]
DEEPSEEK_URL = config["deepseek"]["base_url"]

# ========== Rerank 实现 ==========
# 方案 A: 用 Cohere Rerank（推荐，有现成的 LangChain 集成）
# pip install cohere langchain-cohere
#
# from langchain_cohere import CohereRerank
# reranker = CohereRerank(top_n=2, model="rerank-v3.5")
# 然后在管道中:
#   {"context": retriever | reranker, "input": RunnablePassthrough()}
#
# 方案 B: 用智谱/百炼的 Rerank API（国内可用）
# 方案 C: 自建简单 Rerank（下面演示）


def simple_rerank(query: str, docs: list[Document], top_k: int = 2) -> list[Document]:
    """
    简易 Rerank: 用 LLM 对每个文档打分，取最高分。
    生产环境建议用专门的 Rerank 模型（如 BGE-Reranker、Cohere Rerank）。
    """
    if len(docs) <= top_k:
        return docs  # 候选不足，直接返回

    llm = ChatOpenAI(
        model="deepseek-v4-pro",
        temperature=0,
        api_key=DEEPSEEK_KEY,
        base_url=DEEPSEEK_URL,
    )

    rerank_prompt = PromptTemplate.from_template(
        """你是一个文档相关性评分器。请对以下每份文档与查询的相关性打分（1-5分）。
查询: {query}

文档列表:
{doc_list}

请按以下 JSON 格式返回，只返回 JSON，不要其他内容:
[{{"index": 0, "score": 5}}, {{"index": 1, "score": 3}}]
"""
    )

    doc_list = "\n---\n".join(
        f"[文档{i}]\n{d.page_content[:300]}" for i, d in enumerate(docs)
    )

    response = llm.invoke(rerank_prompt.format(query=query, doc_list=doc_list))
    import re
    import json

    # 提取 JSON
    json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
    if json_match:
        scores = json.loads(json_match.group())
        scores.sort(key=lambda x: x["score"], reverse=True)
        top_indices = [s["index"] for s in scores[:top_k]]
        return [docs[i] for i in top_indices if i < len(docs)]

    return docs[:top_k]


# ========== 带 Rerank 的 RAG 管道 ==========
def build_rag_with_rerank():
    """构建带 Rerank 的 RAG 管道"""

    # 1. 加载文档
    docs = TextLoader("doc/kkfile.md", encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 2. 向量库
    embeddings = BigModelEmbeddingFunc(api_key=EMBEDDING_KEY)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./rag_knowledge_base")
    # 增大 k，召回更多候选供 Rerank 筛选
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 3. 提示词
    prompt = PromptTemplate.from_template(
        """请严格依据下面参考资料回答用户问题，给出回答的依据文件，不要编造内容：
参考资料：{context}
用户问题：{input}
"""
    )

    # 4. LLM
    llm = ChatOpenAI(
        model="deepseek-v4-pro",
        temperature=0.7,
        api_key=DEEPSEEK_KEY,
        base_url=DEEPSEEK_URL,
    )

    # 5. 构建管道: 检索(大k) → Rerank(小n) → LLM
    def rerank_step(inputs):
        """inputs 是 dict: {"context": [Document...], "input": str}"""
        docs = inputs["context"]
        query = inputs["input"]
        return {"context": simple_rerank(query, docs, top_k=2), "input": query}

    rag_chain_with_rerank = (
        rerank_step  # 先做 rerank
        | {"context": RunnableLambda(lambda x: x["context"]), "input": RunnablePassthrough()}  # 透传
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain_with_rerank


# ========== 对比实验 ==========
def compare_with_without_rerank():
    """对比有/无 Rerank 的检索结果"""

    docs = TextLoader("doc/kkfile.md", encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = BigModelEmbeddingFunc(api_key=EMBEDDING_KEY)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./rag_knowledge_base")

    # 不带 Rerank: k=3
    retriever_no_rerank = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 带 Rerank: k=10 召回, top_n=2 精排
    retriever_with_rerank = vectorstore.as_retriever(search_kwargs={"k": 10})

    query = "kkFileView支持哪些3D模型格式？"

    print(f"查询: {query}")
    print(f"\n--- 无 Rerank (Top-3) ---")
    docs_no_rerank = retriever_no_rerank.invoke(query)
    for i, d in enumerate(docs_no_rerank):
        print(f"  [{i+1}] {d.page_content[:100]}...")

    print(f"\n--- 有 Rerank (Top-10 → Top-2) ---")
    docs_with_rerank = retriever_with_rerank.invoke(query)
    reranked = simple_rerank(query, docs_with_rerank, top_k=2)
    for i, d in enumerate(reranked):
        print(f"  [{i+1}] {d.page_content[:100]}...")

    print("\n结论: Rerank 过滤掉了不相关的候选，让 LLM 只看最相关的 2 条")


if __name__ == "__main__":
    print("=" * 60)
    print("Rerank 重排序实验")
    print("=" * 60)
    compare_with_without_rerank()
