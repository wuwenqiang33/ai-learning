"""
Step 09 — 查询改写（Query Rewriting）（Day 14 进阶）

功能：让用户的问题更适合检索。
用户的问题往往模糊、简略或有歧义。
查询改写通过 LLM 对原始问题做扩展、澄清、关键词提取，
让检索结果更精准。

三种改写策略：
1. Query Expansion — 扩展关键词（"kkFileView" → "kkFileView 文件预览 在线预览"）
2. Step-back Prompting — 后退一步（"v5.0.0新功能" → "kkFileView版本更新"）
3. Hypothetical Questions — 假设性问题（生成可能的变体问法）

运行: python steps/step09_query_rewrite.py
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

PG_CONNECTION = (
    "postgresql://root:rootPassword@localhost:5432/rag_db"
)
COLLECTION_NAME = "kkfile_documents"

print("=" * 60)
print("Step 09 — 查询改写（Query Rewriting）")
print("=" * 60)

# ── 1. 加载文档 ──
print("\n[1] 加载文档...")
DOC_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "kkfile.md")
docs = TextLoader(DOC_PATH, encoding="utf-8").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"    加载了 {len(docs)} 篇文档，分成 {len(chunks)} 个块")

# ── 2. 构建检索器 ──
print("[2] 构建检索器...")
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
    search_kwargs={"k": 5},
)

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# ── 3. 构建 LLM（用于查询改写） ──
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    temperature=0,
    api_key=DEEPSEEK_KEY,
    base_url=DEEPSEEK_URL,
)

# ── 4. 三种查询改写策略 ──

# 4a. Query Expansion — 扩展关键词
expand_prompt = PromptTemplate.from_template(
    "你是一个搜索优化助手。用户想查询关于 kkFileView 文档的信息。\n"
    "原始问题: {query}\n\n"
    "请生成 3-5 个相关的扩展查询，帮助更全面地检索文档。\n"
    "扩展查询应该包含：同义词、近义词、相关术语。\n\n"
    "只返回查询列表，每行一个，不要其他内容：\n"
)

def expand_query(query: str) -> list[str]:
    """Query Expansion：扩展关键词"""
    response = llm.invoke(expand_prompt.format(query=query))
    lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
    # 清理前缀（如 "1. " "2. "）
    cleaned = []
    for line in lines:
        import re
        line = re.sub(r"^\d+\.\s*", "", line)
        if line:
            cleaned.append(line)
    return cleaned[:5] if cleaned else [query]

# 4b. Step-back Prompting — 后退一步
stepback_prompt = PromptTemplate.from_template(
    "你是一个搜索优化助手。用户想查询关于 kkFileView 文档的信息。\n"
    "原始问题: {query}\n\n"
    "请生成一个更宽泛、更通用的问题，帮助检索更广泛的相关内容。\n"
    "例如：\n"
    "  - 'v5.0.0新功能' → 'kkFileView版本更新'\n"
    "  - '支持哪些3D格式' → 'kkFileView支持的格式'\n\n"
    "只返回宽泛问题，不要其他内容：\n"
)

def stepback_query(query: str) -> str:
    """Step-back：生成更宽泛的问题"""
    response = llm.invoke(stepback_prompt.format(query=query))
    return response.content.strip()

# 4c. 假设性问题
hypothetical_prompt = PromptTemplate.from_template(
    "你是一个搜索优化助手。用户想查询关于 kkFileView 文档的信息。\n"
    "原始问题: {query}\n\n"
    "请生成 2-3 个用户可能用不同方式表达的相同意图的问题。\n"
    "只返回问题列表，每行一个，不要其他内容：\n"
)

def hypothetical_questions(query: str) -> list[str]:
    """假设性问题：生成等效问法"""
    response = llm.invoke(hypothetical_prompt.format(query=query))
    lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
    cleaned = []
    for line in lines:
        import re
        line = re.sub(r"^\d+\.\s*", "", line)
        if line:
            cleaned.append(line)
    return cleaned[:3] if cleaned else [query]


# ── 5. 改写策略对比 ──
print("\n[3] 查询改写策略对比...")
test_queries = [
    "kkFileView支持哪些文档格式？",
    "v5.0.0版本有什么新功能？",
    "官网地址是什么？",
]

for query in test_queries:
    print(f"\n  原始查询: {query}")

    # Query Expansion
    expanded = expand_query(query)
    print(f"  扩展查询 ({len(expanded)} 条):")
    for eq in expanded:
        print(f"    - {eq}")

    # Step-back
    stepback = stepback_query(query)
    print(f"  后退查询: {stepback}")

    # Hypothetical
    hypothetical = hypothetical_questions(query)
    print(f"  假设问题 ({len(hypothetical)} 条):")
    for hq in hypothetical:
        print(f"    - {hq}")


# ── 6. 用改写后的查询做检索 ──
print("\n[4] 改写后检索效果对比...")

def retrieve_with_expansion(query: str) -> list:
    """用扩展查询做检索，合并结果"""
    expanded = expand_query(query)
    all_docs = []
    seen = set()

    # 对每个扩展查询都做向量 + BM25 检索
    for exp_q in expanded:
        for retriever in [vector_retriever, bm25_retriever]:
            results = retriever.invoke(exp_q)
            for doc in results:
                key = hash(doc.page_content)
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)

    return all_docs[:5]


def retrieve_with_stepback(query: str) -> list:
    """用后退查询做检索"""
    sb_query = stepback_query(query)
    results = []
    for q in [query, sb_query]:
        for retriever in [vector_retriever, bm25_retriever]:
            results.extend(retriever.invoke(q))
    # 去重
    seen = set()
    unique = []
    for doc in results:
        key = hash(doc.page_content)
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique[:5]


for query in test_queries[:2]:
    print(f"\n  查询: {query}")

    # 原始检索
    orig_results = vector_retriever.invoke(query)
    print(f"    原始检索: {len(orig_results)} 条")
    if orig_results:
        print(f"      首条: {orig_results[0].page_content[:80]}...")

    # 扩展检索
    exp_results = retrieve_with_expansion(query)
    print(f"    扩展检索: {len(exp_results)} 条")
    if exp_results:
        print(f"      首条: {exp_results[0].page_content[:80]}...")

    # 后退检索
    sb_results = retrieve_with_stepback(query)
    print(f"    后退检索: {len(sb_results)} 条")
    if sb_results:
        print(f"      首条: {sb_results[0].page_content[:80]}...")


# ── 7. 构建查询改写 RAG 管道 ──
print("\n[5] 构建查询改写 RAG 管道...")

prompt = PromptTemplate.from_template(
    "请严格依据下面参考资料回答用户问题，给出回答的依据文件，不要编造内容：\n"
    "参考资料：{context}\n"
    "用户问题：{input}"
)

def rewrite_step(x):
    """查询改写步骤，接收 dict {"input": query_str}"""
    query = x["input"]
    docs = retrieve_with_expansion(query)
    return {"context": "\n\n".join(d.page_content for d in docs), "input": query}

rag_chain_rewrite = (
    RunnableLambda(lambda x: {"input": x})
    | rewrite_step
    | prompt
    | llm
    | StrOutputParser()
)

# ── 8. 问答测试 ──
print("\n[6] 查询改写 RAG 问答测试...")
for q in test_queries:
    print(f"\n  Q: {q}")
    result = rag_chain_rewrite.invoke(q)
    print(f"  A: {result[:200]}...")

print("\n" + "=" * 60)
print("Step 09 完成 — 查询改写让检索更精准！")
print("=" * 60)
