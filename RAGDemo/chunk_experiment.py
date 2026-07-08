# Chunking 策略实验脚本

"""
对比不同 chunk_size / chunk_overlap 对检索结果的影响。
运行方式: python chunk_experiment.py
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from BigModelEmbedding import BigModelEmbeddingFunc
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import json

# ========== 配置 ==========
with open("config_bigmodel.json", "r", encoding="utf-8") as f:
    config = json.load(f)

EMBEDDING_KEY = config["embedding"]["api_key"]
DEEPSEEK_KEY = config["deepseek"]["api_key"]
DEEPSEEK_URL = config["deepseek"]["base_url"]

DOC_PATH = os.path.join(os.path.dirname(__file__), "doc", "kkfile.md")
TEST_QUERIES = [
    "kkFileView的官网地址？",
    "支持哪些3D模型格式？",
    "v5.0.0版本修复了哪些问题？",
]

# ========== 实验配置 ==========
# 每组: (splitter_name, splitter_instance)
splitters = [
    ("RecursiveCharacterTextSplitter (500/50)",
     RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)),
    ("RecursiveCharacterTextSplitter (800/100)",
     RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)),
    ("RecursiveCharacterTextSplitter (1000/200)",
     RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)),
    ("CharacterTextSplitter (500)",
     CharacterTextSplitter(chunk_size=500, chunk_overlap=50)),
]


def load_and_split(splitter, name):
    """加载文档并用指定 splitter 分割，返回 chunks 统计"""
    docs = TextLoader(DOC_PATH, encoding="utf-8").load()
    chunks = splitter.split_documents(docs)

    # 统计信息
    avg_len = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
    print(f"\n{'='*60}")
    print(f"策略: {name}")
    print(f"  分块数: {len(chunks)}")
    print(f"  平均长度: {avg_len:.0f} 字符")
    print(f"  最小长度: {min(len(c.page_content) for c in chunks)}")
    print(f"  最大长度: {max(len(c.page_content) for c in chunks)}")

    # 展示第一个 chunk 的内容预览
    if chunks:
        print(f"  首个 chunk 预览: {chunks[0].page_content[:100]}...")

    return chunks


def evaluate_chunking(chunks, name, embeddings):
    """用简单检索+LLM判断来评估不同策略"""
    db_path = f"./experiment_db_{name.replace('/', '_').replace(' ', '_')}"
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)

    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="deepseek-v4-pro",
        temperature=0,
        api_key=DEEPSEEK_KEY,
        base_url=DEEPSEEK_URL,
    )

    print(f"\n  各查询的 Top-3 检索结果:")
    for query in TEST_QUERIES:
        results = retriever.invoke(query)
        print(f"\n  查询: '{query}'")
        for i, doc in enumerate(results):
            preview = doc.page_content[:120].replace('\n', ' ')
            print(f"    [{i+1}] {preview}...")

    # 清理
    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Chunking 策略实验")
    print("=" * 60)

    embeddings = BigModelEmbeddingFunc(api_key=EMBEDDING_KEY)

    for name, splitter in splitters:
        chunks = load_and_split(splitter, name)
        evaluate_chunking(chunks, name, embeddings)

    print("\n" + "=" * 60)
    print("实验完成。建议:")
    print("  - chunk_size 太小 → 上下文碎片化，LLM 难以连贯理解")
    print("  - chunk_size 太大 → 检索精度下降，混入无关信息")
    print("  - chunk_overlap 太小 → 相邻块语义断裂")
    print("  - chunk_overlap 太大 → 冗余增加，浪费 token")
    print("  - 经验值: chunk_size=500~1000, overlap=50~200")
    print("=" * 60)
