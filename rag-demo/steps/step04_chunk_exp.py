"""
Step 04 — Chunking 策略实验（Day 8 进阶）

功能：对比不同 chunk_size / chunk_overlap 对检索结果的影响。
      这是优化 RAG 系统的关键一步。

运行: python steps/step04_chunk_exp.py
"""

import sys, os

from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.append(str(Path(__file__).parent.parent))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from lib.embedding import BigModelEmbeddingFunc
from lib.config import load_config

config = load_config()
EMBEDDING_KEY = config["embedding"]["api_key"]
DEEPSEEK_KEY = config["deepseek"]["api_key"]
DEEPSEEK_URL = config["deepseek"]["base_url"]
DEEPSEEK_MODEL = config["deepseek"]["model"]
DOC_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "kkfile.md")

TEST_QUERIES = [
    "kkFileView的官网地址？",
    "支持哪些3D模型格式？",
    "v5.0.0版本修复了哪些问题？",
]

# ── 实验配置 ──
splitters = [
    ("500_50", RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)),
    ("800_100", RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)),
    ("1000_200", RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)),
]

print("=" * 60)
print("Step 04 — Chunking 策略实验")
print("=" * 60)

embeddings = BigModelEmbeddingFunc(api_key=EMBEDDING_KEY)

for name, splitter in splitters:
    # 加载 + 分割
    docs = TextLoader(DOC_PATH, encoding="utf-8").load()
    chunks = splitter.split_documents(docs)
    avg_len = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0

    print(f"\n{'─' * 50}")
    print(f"策略: {name}  |  块数: {len(chunks)}  |  平均长度: {avg_len:.0f}")
    print(f"{'─' * 50}")

    # 创建临时向量库
    db_path = os.path.join(
        os.path.dirname(__file__), "..", "output", f"exp_{name.replace('_', '')}"
    )
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 构建管道
    prompt = PromptTemplate.from_template(
        "请严格依据下面参考资料回答用户问题，给出回答的依据文件，不要编造内容：\n"
        "参考资料：{context}\n"
        "用户问题：{input}"
    )
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL, temperature=0,
        api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_URL,
    )
    rag_chain = {"context": retriever, "input": RunnablePassthrough()} | prompt | llm | StrOutputParser()

    # 测试查询
    for query in TEST_QUERIES:
        result = rag_chain.invoke(query)
        print(f"  Q: {query}")
        print(f"  A: {result[:120]}...")

print("\n" + "=" * 60)
print("实验完成 — 经验值: chunk_size=500~1000, overlap=50~200")
print("=" * 60)
